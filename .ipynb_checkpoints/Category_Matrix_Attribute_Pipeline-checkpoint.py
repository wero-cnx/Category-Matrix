# %% [markdown]
# # Category Matrix – Attribute Assignment Pipeline
#
# **What this notebook does**
# 1. Reads the six source files and the PDF with the specialists' examples (`Category Matrix/SRC`).
# 2. Selects all products of the six sub-categories **Vase, Esstisch, Loungesessel, Regalsysteme, Akkuleuchten, Tassen**.
# 3. Assigns *every* attribute defined for the sub-category in `category_attribute_values_vertical.csv`.
#    Evidence: structured attributes → product text → dimensions → **product photos (embeddings of the cropped product)** → **PDF examples (few-shot similarity)**.
# 4. Aggregates sales, validates everything and writes the result files to `Category Matrix/RES`.
#
# **How to run:** edit only *Section 1a / 1b*, then `Kernel → Restart & Run All`. Everything else is rule-free pipeline code.
# Re-runs reuse cached images and embeddings (`RES/cache_*`), so they are fast.
#
# | Section | Content |
# |---|---|
# | 1 | Configuration (1a settings, 1b domain rules / vocabularies) |
# | 2 | Load & validate files, schema discovery, data quality |
# | 3 | Allowed attributes & values |
# | 4 | Select the six sub-categories |
# | 5 | Join products + item attributes + image links (+ your extra columns) |
# | 6 | Extract examples & rules from `Designstyles.pdf` |
# | 7 | Download, cache, validate images; find & crop the product |
# | 8 | Embeddings, PDF few-shot model, zero-shot image model |
# | 9 | Attribute assignment, confidence & evidence |
# | 10 | Sales loading & aggregation |
# | 11 | Validation checks |
# | 12 | Exports, summary tables, README |

# %% [markdown]
# ## 1a. Configuration  (the only place you normally edit)
# Paths, source-file patterns, column mappings, target sub-categories, sales settings, image + model settings.
# *Optional* packages are marked – the notebook falls back automatically if they are missing.

# %%
# ---- Imports -------------------------------------------------------------------------------
import os, re, io, json, math, time, shutil, pickle, hashlib, warnings, unicodedata, colorsys, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:                       # running as a plain script
    def display(x): print(x)

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Paths ---------------------------------------------------------------------------------
PROJECT_DIR_OVERRIDE = None             # e.g. r"C:\Work\Category Matrix"  (None = auto-detect)

def _find_project_dir():
    if PROJECT_DIR_OVERRIDE: return Path(PROJECT_DIR_OVERRIDE)
    cwd = Path.cwd()
    if (cwd / "SRC").exists(): return cwd
    if (cwd / "Category Matrix" / "SRC").exists(): return cwd / "Category Matrix"
    return cwd

PROJECT_DIR = _find_project_dir()
SRC_DIR = PROJECT_DIR / "SRC"
RES_DIR = PROJECT_DIR / "RES"
DIRS = {
    "res": RES_DIR,
    "cache_images_raw":  RES_DIR / "cache_images" / "raw",
    "cache_images_crop": RES_DIR / "cache_images" / "crops",
    "cache_embeddings":  RES_DIR / "cache_embeddings",
    "pdf_examples":      RES_DIR / "intermediate" / "pdf_examples",
    "intermediate":      RES_DIR / "intermediate",
    "review_sheets":     RES_DIR / "intermediate" / "review_sheets",
}
for _d in DIRS.values(): _d.mkdir(parents=True, exist_ok=True)

# ---- Main configuration --------------------------------------------------------------------
CFG = {
  # ---------- source files: matched by *normalised* file name (lower case, letters+digits only)
  "files": {
    "products":        {"patterns": ["dwhpresentselect", "dwhpresent"],            "required": True},
    "item_attributes": {"patterns": ["itemattribute"],                              "required": False},
    "media":           {"patterns": ["itemmedia"],                                  "required": False},
    "sales":           {"patterns": ["grouporders", "orderssales", "ordersandsales"], "required": False},
    "allowed_values":  {"patterns": ["categoryattributevalues"],                    "required": True},
    "pdf":             {"patterns": ["designstyles"], "extensions": [".pdf"],       "required": False},
  },

  # ---------- column mappings: logical name -> list of accepted header names (matched case/space/separator-insensitive)
  "columns": {
    "products": {
      "product_id":      ["ItemId", "Item Id", "Item_ID", "ArticleId", "Artikelnummer"],
      "product_name":    ["Name", "Product name", "Bezeichnung", "Description"],
      "manufacturer_id": ["ManufacturerId", "Manufacturer"],
      "top_category":    ["ShopifyTopCategory"],
      "main_category":   ["ShopifyMainCategory"],
      "sub_category":    ["ShopifySubCategory", "SubCategory", "Unterkategorie"],
      "length":          ["Length", "Laenge", "Länge"],
      "depth":           ["Depth", "Tiefe"],
      "height":          ["Height", "Hoehe", "Höhe"],
      "weight":          ["Weight", "Gewicht"],
      "variant":         ["Variant", "Variante"],
      "color_name":      ["ColorName", "Colour", "Color", "Farbe"],
      "material":        ["Material"],
    },
    "media":           {"product_id": ["ItemId"], "media_id": ["MediaId"], "order": ["Order", "Position"],
                        "url": ["PartialPath", "Url", "ImageUrl", "Path"]},
    "item_attributes": {"product_id": ["ItemId"], "set": ["AttributeSetName"],
                        "name": ["AttributeName"], "value": ["AttributeValue"]},
    "allowed_values":  {"category": ["Category", "Kategorie"], "attribute": ["Attribute", "Attribut"],
                        "value": ["Value", "Wert"]},
  },
  "units": {"dimension_divisor_to_cm": 10.0},    # dwh Length/Depth/Height are in mm -> cm

  # ---------- target sub-categories ------------------------------------------------------------
  # key = name as you call it; matrix_category = name used in category_attribute_values_vertical.csv
  # (auto-resolved by name containment, e.g. "Esstisch" -> "Design-Esstisch"); dwh_values = values in the product file.
  # rule_key links to the rules in 1b (prompts, keywords, PDF pages ...).
  "targets": {
    "Vase":         {"rule_key": "vase",          "dwh_values": ["Vasen"],        "matrix_category": None},
    "Esstisch":     {"rule_key": "dining_table",  "dwh_values": ["Esstische"],    "matrix_category": None},
    "Loungesessel": {"rule_key": "lounge_chair",  "dwh_values": ["Loungesessel"], "matrix_category": None},
    "Regalsysteme": {"rule_key": "shelving",      "dwh_values": ["Regalsysteme"], "matrix_category": None},
    "Akkuleuchten": {"rule_key": "portable_lamp", "dwh_values": ["Akkuleuchten"], "matrix_category": None},
    "Tassen":       {"rule_key": "cup",           "dwh_values": ["Tassen"],       "matrix_category": None},
  },
  "include_products_by_item_attribute_category": False,  # True: also select products whose ItemAttribute 'Shopify Sub Category' matches

  # ---------- YOUR EXTRA INPUT COLUMNS (price, tier, ...) ----------------------------------------
  # Any column found here is normalised and carried to ALL outputs. Columns not listed are also carried
  # automatically when carry_unmapped_product_columns=True. To use a column as *evidence* for an attribute see
  # ATTRIBUTE_FROM_COLUMN in 1b.
  "extra_product_fields": {
    "price": {"candidates": ["Price", "Preis", "RetailPrice", "Retail price", "VK", "Verkaufspreis", "UVP", "SalesPrice"], "type": "number"},
    "tier":  {"candidates": ["Tier", "Tier category", "TierCategory", "Price tier", "Preisklasse", "Segment"],               "type": "text"},
  },
  "carry_unmapped_product_columns": True,
  # Additional files with one row per product (e.g. a price list). Joined on the product key, row count is protected.
  "extra_sources": [
    # {"name": "price_list", "patterns": ["pricelist", "preise"], "key_candidates": ["ItemId"],
    #  "keep_columns": None, "prefix": "pl_", "numeric_columns": ["Price"]},
  ],

  # ---------- sales -----------------------------------------------------------------------------
  "sales": {
    "period_start": "2025-11-01", "period_end": "2026-10-31",        # inclusive
    "dayfirst": True,
    "currency": "EUR", "currency_verified": False,                    # not stated in the file -> assumption (connox.de = EUR)
    "columns": {                                                      # logical name -> accepted headers
      "product_id":   ["Item (group)", "ItemId", "Item", "Artikel"],
      "product_name": ["Unnamed: 1", "Item name", "Name", "Bezeichnung"],
      "item_type":    ["Item type (CNX 2.0)", "Item type"],
      "date":         ["Date", "Order date", "Datum", "Bestelldatum", "Month", "Monat", "Period", "Posting date"],
      "order_id":     ["Order", "Order id", "Order number", "Bestellnummer", "Auftrag"],
      "units_gross":  ["Order item qty (group)"],
      "revenue_gross": ["Net order intake I (group)"],
      "units_net":    ["NOI II item qty (group)"],
      "revenue_net":  ["Net order intake II (group)"],
      "stock_available": ["Stock qty - available"],
    },
    "primary_units": "units_net", "primary_revenue": "revenue_net",     # used for ranking and the headline columns
    "rank_by": "units",                                               # 'units' or 'revenue'
  },

  # ---------- images ----------------------------------------------------------------------------
  "images": {
    "download": True,                 # False = use only what is already cached
    "max_images_per_product": 3,      # best-ranked candidate images per product
    "max_products": None,             # e.g. 50 for a quick test run
    "timeout_connect": 5, "timeout_read": 20, "retries": 2, "backoff": 0.6,
    "max_workers": 8, "max_bytes": 15_000_000, "min_side_px": 80,
    "retry_failed_downloads": False,  # True = try URLs again that failed in an earlier run
    "user_agent": "CategoryMatrixBot/1.0 (internal portfolio analysis)",
    "auth_header_env": None,          # name of an ENVIRONMENT VARIABLE holding e.g. 'Bearer xyz' - never write secrets here
    "url_prefix": "",                 # prepended when a link has no http(s):// scheme
    "resize_query": None,             # e.g. "width=800" for Shopify CDNs (saves bandwidth)
    "lifestyle_url_words": ["situation", "ambiente", "lifestyle", "detail", "scene", "mood", "room", "interior"],
    "crop": {"strategies": ["detector", "rembg", "background"],       # tried in this order for non-studio photos
             "margin": 0.06, "composite_on_white": True, "target_px": 448},
  },

  # ---------- models ----------------------------------------------------------------------------
  # backend: 'auto' -> transformers if installed, else open_clip, else 'classic' (colour+shape features, no text model)
  "models": {
    "backend": "auto",                # auto | transformers | open_clip | classic
    # Primary joint image-text model (OPTIONAL: pip install torch transformers sentencepiece protobuf)
    # fast default: "google/siglip-base-patch16-224"; stronger but slower: "google/siglip-so400m-patch14-384"
    "primary_hf_id": "google/siglip-base-patch16-224",
    "open_clip_name": "ViT-B-16-SigLIP", "open_clip_pretrained": "webli",
    # Second, shape-focused vision model used for similarity to the PDF examples (OPTIONAL)
    "secondary": {"enabled": True, "hf_id": "facebook/dinov2-small", "weight": 0.35},
    # Open-vocabulary detector to find the product in lifestyle photos (OPTIONAL, heavier)
    "detector": {"enabled": True, "hf_id": "google/owlv2-base-patch16-ensemble", "min_score": 0.10},
    "device": "auto", "batch_size": 16, "local_files_only": False,
    "zero_shot_temperature": 100.0,   # CLIP-style logit scale for the softmax over prompts
    "use_rembg": True,                # OPTIONAL (pip install rembg onnxruntime): cleaner product masks
    "classic_feature_size": 64,
  },

  # ---------- PDF -------------------------------------------------------------------------------
  "pdf": {
    # page ranges (1-based, inclusive) of each product group inside Designstyles.pdf
    "page_ranges": {"dining_table": [1, 4], "lounge_chair": [5, 8], "cup": [9, 12], "shelving": [13, 14], "portable_lamp": [15, 18]},
    # page heading text (normalised) -> style label used in the allowed-values file
    "style_headings": {"minimal": "Minimalistic", "minimalistisch": "Minimalistic", "minimalistic": "Minimalistic",
                       "fashionable": "Fashionable", "skulptural": "Sculptural", "sculptural": "Sculptural",
                       "classic": "Classic", "klassisch": "Classic"},
    "min_example_side_px": 60,
    "near_duplicate_similarity": {"deep": 0.90, "classic": 0.995},   # product ~ identical to a PDF example
    "knn_k": 3, "fewshot_mix_lr": 0.5,                               # weight of logistic regression vs kNN
    "reliability_floor": 0.35, "reliability_cap": 0.90,              # PDF model reliability = LOO accuracy clipped to this range
    "transfer_to_categories_without_examples": True,                # e.g. Vase: use style examples of other groups (mean-centred)
    "transfer_reliability_factor": 0.6,
  },

  # ---------- assignment / confidence ------------------------------------------------------------
  "assignment": {
    "sentinel_not_determinable": "Not determinable",
    "sentinel_none_detected":    "None detected",     # multi-value attributes (e.g. Add ons) with no value present
    "multi_value_attributes": {"lounge_chair": ["Add ons"]},
    "multi_value_threshold": 0.50,
    "multi_value_separator": " | ",
    "conflict_penalty": 0.75,                         # confidence = best score - penalty * runner-up score
    "levels": {"High": 0.75, "Medium": 0.50, "Low": 0.30},   # below Low -> Not determinable
    "style_cap_without_pdf_near_duplicate": "Medium", # subjective attribute: High only if near-identical to a PDF example
    "style_cap_without_own_pdf_examples": "Low",      # e.g. Vase (no vase pages in the PDF)
    "lifestyle_image_penalty": 0.60,                  # reliability factor when only a lifestyle photo is available
    "lounge_seat_width_estimate": True,               # estimate seat width from overall width (Low confidence)
    "reliability": {                                  # how much each evidence type is trusted (0-1)
      "product_attribute": 0.95, "column_map": 0.95, "product_text": 0.85, "dimensions": 0.70,
      "image_colour": 0.80, "image_zeroshot": 0.65, "style_zeroshot": 0.45, "style_text_hint": 0.40,
      "generic_text": 0.55, "estimate": 0.35,
    },
  },
  "validation": {"strict": False,       # True = raise an error when a *structural* check fails
                 "max_not_determinable_share_warn": 0.50},
}

# Make the config auditable: save it next to the results
def _json_safe(o):
    if isinstance(o, dict): return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
    return o if isinstance(o, (str, int, float, bool)) or o is None else str(o)
(RES_DIR / "run_config.json").write_text(json.dumps(_json_safe(CFG), indent=2, ensure_ascii=False), encoding="utf-8")
print("Project folder:", PROJECT_DIR, "\nSRC:", SRC_DIR, "\nRES:", RES_DIR)

# %% [markdown]
# ## 1b. Domain rules & vocabularies (advanced – tune the logic here, not in the pipeline code)
# * **PIPELINE** – which evidence providers run for which attribute (new attribute in the allowed-values file? add a line).
# * **IMAGE_PROMPTS / STYLE_PROMPTS** – text descriptions the image model compares the *cropped product* with.
#   The style prompts were written from the examples in `Designstyles.pdf` (see the interpretation table in Section 6).
# * **KEYWORD_RULES** – regular expressions for German/English product text (matched after folding ä→ae, ß→ss, accents removed).
# * **COLOUR_LEXICON / COLOUR_VALUE_MAP** – colour words → colour groups → allowed values.
# * **DIMENSION rules** – seats from table size, vase form from proportions, seat width bins.
# * **ABSENCE_DEFAULTS** – value assumed when no evidence contradicts it (with its trust level). Remove a line to get "Not determinable" instead.
# * **ATTRIBUTE_FROM_COLUMN** – use *your new columns* (price, tier ...) to fill attributes.

# %%
NOUNS = {"vase": "a vase", "dining_table": "a dining table", "lounge_chair": "a lounge armchair",
         "shelving": "a shelving unit", "portable_lamp": "a portable cordless table lamp", "cup": "a cup or mug"}

DETECTOR_QUERIES = {   # used by the open-vocabulary detector to find the product in lifestyle photos
    "vase": ["a vase"], "dining_table": ["a dining table", "a table"], "lounge_chair": ["an armchair", "a lounge chair"],
    "shelving": ["a shelf", "a bookcase", "a shelving unit"], "portable_lamp": ["a lamp", "a table lamp", "a lantern"],
    "cup": ["a cup", "a mug"],
}

# ---- Which providers run per attribute ---------------------------------------------------------
_STYLE = ["style_fewshot", "style_zeroshot", "style_text_hint"]
PIPELINE = {
  "vase": {
    "Design style": _STYLE,
    "Material": ["keywords", "image_zeroshot"],
    "Form & flower fit": ["vase_form", "keywords", "image_zeroshot"],
    "Colour & finish": ["colour_text", "colour_image", "image_zeroshot"],
  },
  "dining_table": {
    "Design style": _STYLE,
    "Size (seats)": ["table_seats"],
    "Material & look (table top)": ["keywords", "image_zeroshot"],
    "Functionality": ["keywords", "image_zeroshot"],
  },
  "lounge_chair": {
    "Design style": _STYLE,
    "Size (seat width)": ["lounge_seat_width"],
    "Seat Material & look": ["keywords", "image_zeroshot"],
    "Frame Material & look": ["keywords", "image_zeroshot"],
    "Material & look": ["keywords", "image_zeroshot"],
    "Frame Functionality": ["keywords", "image_zeroshot"],
    "Add ons": ["keywords", "image_zeroshot"],
  },
  "shelving": {
    "Material & look": ["keywords", "colour_image", "image_zeroshot"],
    "Functionality": ["keywords", "image_zeroshot"],
    "Colour Group": ["colour_text", "colour_image"],
  },
  "portable_lamp": {
    "Design style": _STYLE,
    "Colour Group": ["colour_text", "colour_image"],
    "Material & look": ["keywords", "image_zeroshot"],
    "Light control": ["keywords"],
    "Environment": ["keywords"],
  },
  "cup": {
    "Design style": _STYLE,
    "Material": ["keywords", "image_zeroshot"],
    "Colour & finish": ["colour_text", "colour_image"],
  },
}
DEFAULT_PIPELINE = ["column_map", "generic_text"]   # for attributes you add to the allowed-values file and have no rule for yet

# ---- Design style: prompts derived from the PDF examples (Section 6 shows the interpretation) ----
STYLE_PROMPTS = {
  "dining_table": {
    "Minimalistic": ["a minimalist dining table with simple clean lines, slim straight legs and a plain light wood, white or black top",
                     "a Scandinavian minimal table in neutral colours"],
    "Sculptural":   ["a sculptural dining table with an unusual bold base such as a trestle, pedestal, tripod, wire frame or thick block legs",
                     "a table with a marble or travertine top and an architectural base"],
    "Fashionable":  ["a fashionable dining table in saturated colours such as blue, pink or magenta with a glossy lacquered finish",
                     "a colourful playful contemporary table"],
    "Classic":      ["a classic dining table with a warm wooden top and traditional tapered round legs",
                     "a traditional oval or round extendable oak table in mid-century style"],
  },
  "lounge_chair": {
    "Minimalistic": ["a minimalist lounge chair with a restrained simple shape, muted colours and slim legs or a swivel base",
                     "a simple modern armchair with straight lines"],
    "Sculptural":   ["a sculptural armchair with a bold curved silhouette, chunky rounded forms or an expressive carved structure",
                     "an artistic statement lounge chair such as an egg chair or a boucle blob chair"],
    "Fashionable":  ["a fashionable lounge chair in a bright colour with soft puffy upholstery in a trendy contemporary shape",
                     "a colourful playful armchair"],
    "Classic":      ["a classic lounge chair with traditional upholstery, wood or leather details, tufting or a wingback in mid-century style",
                     "a timeless iconic armchair like an Eames lounge chair or a wingback chair"],
  },
  "cup": {
    "Minimalistic": ["a minimalist cup with a simple shape, a matte single-colour finish and subdued colours",
                     "a plain clear glass mug or a neutral ceramic cup"],
    "Sculptural":   ["a sculptural cup with an unusual form, for example angular, asymmetric, donut-shaped or hammered silver",
                     "an artistic cup with an expressive handle"],
    "Fashionable":  ["a fashionable cup with bright colours, decorative patterns, drip glaze or a playful handle",
                     "a colourful patterned mug"],
    "Classic":      ["a classic cup with saucer, traditional pattern, thin stripes, floral print or speckled glaze",
                     "a timeless porcelain tea cup with saucer"],
  },
  "portable_lamp": {
    "Minimalistic": ["a minimalist portable lamp with a simple geometric shape in a single muted colour",
                     "a plain mushroom or cylinder shaped cordless table lamp"],
    "Sculptural":   ["a sculptural lamp with an unusual form, faceted, lantern-like, cloud-shaped or a cone with a strong silhouette",
                     "an artistic lamp object"],
    "Fashionable":  ["a fashionable portable lamp in bright saturated colours such as orange, mint, pink or blue",
                     "a colourful playful cordless lamp"],
    "Classic":      ["a classic table lamp with brass or gold metal, a bell or dome shade and a traditional elegant shape",
                     "a timeless lantern lamp with glass and brass details"],
  },
  "shelving": {
    "Minimalistic": ["a minimalist shelving system in white or light wood with a simple grid"],
    "Fashionable":  ["a fashionable colourful shelving unit in bright colours"],
  },
  # No vase pages exist in Designstyles.pdf -> generic descriptions, confidence is capped (cfg: style_cap_without_own_pdf_examples)
  "vase": {
    "Minimalistic": ["a minimalist vase with a simple shape, plain surface and neutral colour"],
    "Sculptural":   ["a sculptural vase with a bold artistic shape that looks like an art object"],
    "Fashionable":  ["a fashionable vase in bright colours or with a playful trendy design"],
    "Classic":      ["a classic vase with a traditional timeless shape"],
  },
}
# Weak text hints (only used when the product text literally says so)
STYLE_TEXT_HINTS = {
  "Minimalistic": [r"minimal", r"schlicht"], "Sculptural": [r"skulptur", r"sculpt"],
  "Fashionable": [r"\bneon\b", r"\bpop\b"], "Classic": [r"klassisch", r"classic\b", r"traditionell", r"vintage", r"retro"],
}
STYLE_TEXT_FIELDS = ["name", "variant", "Printkatalogartikel Text", "Printkatalogartikel Verkürzte Bezeichnung"]

# ---- Image prompts per attribute value (zero-shot on the cropped product) --------------------------
# kind 'single' = pick one value; kind 'binary' = value is (present_prompt, absent_prompt) for multi-value attributes.
IMAGE_PROMPTS = {
  "vase": {
    "Material": {"values": {"Glass": "made of clear or coloured glass", "Ceramic": "made of glazed ceramic or earthenware",
                  "Porcelain": "made of smooth white porcelain", "Mixed materials": "made of two different materials, for example glass with a wooden or metal part",
                  "Metallic finish": "with a shiny metal surface", "Other": "made of an unusual material such as plastic, paper, wood, concrete or rattan"}},
    "Form & flower fit": {"values": {"Round": "with a round bellied shape", "Slender": "that is tall and slender for single stems",
                  "Tubes": "that is a straight cylinder tube", "Floor vase": "that is a very large floor vase for branches",
                  "Sculptural/Objects": "that looks like a sculpture or art object"}},
    "Colour & finish": {"values": {"White/neutral": "in white, cream or beige", "Black/dark": "in black, anthracite or another dark colour",
                  "Transparent/clear": "that is transparent and clear", "Colourful or printed": "in bright colours or with a printed pattern",
                  "Metallic finish": "with a gold, silver, copper or brass metallic finish"}},
  },
  "dining_table": {
    "Material & look (table top)": {"values": {"Light wood top": "with a light natural oak, ash or birch wood top", "Dark wood top": "with a dark walnut or dark stained wood top",
                  "Stone/marble top": "with a marble, travertine or stone top", "Glass top": "with a glass top", "Metal top": "with a metal or steel top"},
                  "reject": ["with a white or coloured laminate top", "with a coloured lacquered top"]},
    "Functionality": {"values": {"Fixed & simple": "that is a plain fixed table", "Extendable": "that is an extendable table with a leaf extension",
                  "Foldable/space-saving": "that is a foldable drop-leaf table", "Wall-mounted/console": "that is a narrow console table or wall-mounted table",
                  "Modular (combinable)": "that is a modular table made of combinable elements"}, "reliability_factor": 0.6},
  },
  "lounge_chair": {
    "Seat Material & look": {"values": {"Leather": "upholstered in leather", "Neutral fabric (grey, beige, white)": "upholstered in neutral grey, beige or white fabric",
                  "Colour or pattern fabric": "upholstered in colourful or patterned fabric", "Fur": "covered with fluffy teddy, shearling or faux fur",
                  "Other": "with a seat made of rattan, plastic, wood or wool"}},
    "Frame Material & look": {"values": {"Textile": "that is fully upholstered with no visible frame or legs", "Wood": "with a visible wooden frame or wooden legs",
                  "Metal": "with a visible metal frame or metal legs", "Plastic": "with a moulded plastic shell", "Other": "with another type of frame such as rattan or cane"}},
    "Material & look": {"values": {"Leather": "upholstered in leather", "Neutral fabric (grey, beige, white)": "upholstered in neutral grey, beige or white fabric",
                  "Colour or pattern fabric": "upholstered in colourful or patterned fabric", "Exposed wood frame": "with an exposed wooden frame and wooden arms",
                  "Metal/outdoor frame": "with a metal frame for outdoor use"}, "reliability_factor": 0.7},
    "Frame Functionality": {"values": {"Fixed frame (no mechanism)": "with a fixed frame and four legs", "Swivel base": "with a swivel star base or rotating pedestal base",
                  "Rocking/gliding base": "with curved rocking or gliding runners", "Adjustable recline": "that is a recliner with an adjustable backrest and footrest"},
                  "reliability_factor": 0.8},
    "Add ons": {"kind": "binary", "values": {
                  "Ottoman": ("shown together with a matching ottoman or footstool", "shown alone without an ottoman"),
                  "Headrest": ("with a headrest or a high wing back", "with a low back and no headrest"),
                  "Armrest": ("with armrests", "without armrests, an armless chair")}, "reliability_factor": 0.8},
  },
  "shelving": {
    "Material & look": {"values": {"Light wood": "made of light natural oak, ash or birch wood", "Dark wood": "made of dark walnut or dark stained wood",
                  "Black/metal frame": "with a black or metal frame", "White/light lacquer": "in a white or light lacquered finish",
                  "Colourful or plastic": "in bright colours or plastic", "Mix": "made of a mix of different materials and colours"}},
    "Functionality": {"values": {"Freestanding single unit": "that is a freestanding single bookcase", "Wall-mounted/floating": "that is a wall-mounted floating shelf",
                  "Enclosed with doors or drawers": "that is a cabinet with doors or drawers", "Ladder/leaning": "that is a ladder shelf leaning against a wall"}},
  },
  "portable_lamp": {
    "Material & look": {"values": {"Ceramic/porcelain": "made of ceramic or porcelain", "Glass": "made of glass", "Metal": "made of metal such as aluminium or brass",
                  "Fabric shade": "with a fabric lampshade", "Plastic/silicone": "made of plastic or soft silicone", "Other": "made of another material such as wood or paper"}},
  },
  "cup": {
    "Material": {"values": {"Fine porcelain": "made of thin white fine porcelain", "Stoneware": "made of handmade stoneware with a matte or speckled glaze",
                  "Glass": "made of glass", "Enamel/metal": "made of enamelled metal", "Other": "made of plastic, bamboo, silicone or wood"}},
  },
}
PROMPT_TEMPLATE = "a product photo of {noun} {desc}"

# ---- Keyword rules on product text ------------------------------------------------------------
# table entries: (value, regex)  or (value, regex, options). value=None only *consumes* text (e.g. 'ohne Türen').
# options: {"unless": [values]} skip if one of these values is found anywhere in the product;  {"rel": 0.7} reliability factor.
# Fields: dwh columns (name, variant, color_name, material) and any ItemAttribute name (e.g. 'Shopify Material').
_NEUTRAL = r"weiss|white|creme|cream|beige|sand|grau|gr[ae]y|taupe|greige|natur\b|elfenbein|ivory|ecru|offwhite|off-white|schwarz|black|anthrazit|nero"
_COLOURFUL = (r"rot\b|red\b|blau|blue|gruen|green|gelb|yellow|orange|pink|rosa|petrol|lila|purple|tuerkis|turquoise|bordeaux|"
              r"braun|brown|cognac|camel|terrakotta|terracotta|rost|rust|senf|mustard|bunt|multi|karo|streifen|muster|pattern|floral")
KEYWORD_RULES = {
  "vase": {
    "Material": {"fields": ["material", "Shopify Material", "variant", "name"], "mixed_value": "Mixed materials",
                 "mixed_fields": ["material", "Shopify Material"],
                 "table": [("Porcelain", r"porzellan|porcelain|bone china|biskuit|bisque"),
                           ("Glass", r"glas(?!ur)|glass|kristall|crystal|borosilikat"),
                           ("Ceramic", r"keramik|ceramic|steingut|steinzeug|terrakotta|terracotta|stoneware|earthenware|fayence|\bton\b"),
                           ("Metallic finish", r"metall|messing|brass|kupfer|copper|edelstahl|stainless|aluminium|\bstahl|steel|zink|bronze|vergoldet|verchromt|nickel|eisen"),
                           ("Other", r"holz|wood|beton|concrete|zement|kunststoff|plastic|polyresin|resin|papier|paper|rattan|bambus|bamboo|kork|marmor|marble|alabaster|wachs|filz|textil")]},
  },
  "dining_table": {
    "Material & look (table top)": {"fields": ["Shopify Material", "material", "Massivholz", "variant", "name", "Shopify Farbe"], "leftmost": True,
        "table": [("Dark wood top", r"(eich|oak|esch|buch|beech|nuss|holz|wood)\w*[\s,/-]*\w*[\s,/-]*(dunkel|schwarz|black|dark|braun|brown|rauch|smoked|wenge)"),
                  ("Dark wood top", r"walnuss|nussbaum|walnut|wenge|mahagoni|mahogany|ebenholz|rauchereiche|raeuchereiche|dark oak|teak"),
                  ("Stone/marble top", r"marmor|marble|travertin|granit|naturstein|stein(?!zeug)|keramik|ceramic|beton|concrete|terrazzo|onyx|dekton|fenix"),
                  ("Glass top", r"glas(?!ur)|glass"),
                  ("Metal top", r"metall|stahl|steel|aluminium|messing|brass|kupfer|chrom|eisen"),
                  ("Light wood top", r"eich|oak|esch|\bash\b|buche|beech|birke|birch|ahorn|maple|kiefer|pine|fichte|spruce|holz|wood|bambus|natur\b|furnier|veneer|sperrholz|plywood")]},
    "Functionality": {"fields": ["name", "variant", "Shopify Ausstattung", "Shopify Größe", "Printkatalogartikel Text"],
        "table": [("Extendable", r"ausziehbar|ausziehtisch|auszug|extendable|extension|erweiterbar|verlaengerbar|einlegeplatte|ansteckplatte|\d{2,3}\s*/\s*\d{2,3}\s*x"),
                  ("Foldable/space-saving", r"klappbar|klapptisch|foldable|folding|faltbar|zusammenklappbar|space[- ]saving|platzsparend|drop[- ]leaf"),
                  ("Wall-mounted/console", r"wandtisch|wandmontage|wall[- ]mounted|konsole|console|wandklapp"),
                  ("Modular (combinable)", r"modular|modulsystem|kombinierbar|combinable|baukasten")]},
  },
  "lounge_chair": {
    "Seat Material & look": {"fields": ["Shopify Sitzbezug", "Shopify Material", "Shopify Sitz", "variant", "name", "Shopify Farbe"],
        "table": [("Fur", r"\bfell|fur\b|teddy|schaffell|lammfell|shearling|webpelz|kunstfell|sheepskin|pluesch|plush"),
                  ("Leather", r"leder|leather|ecoleder|kunstleder|nubuk|cuero|nappa"),
                  ("Colour or pattern fabric", _COLOURFUL, {"unless": ["Leather", "Fur"], "rel": 0.7}),
                  ("Neutral fabric (grey, beige, white)", _NEUTRAL, {"unless": ["Leather", "Fur"], "rel": 0.7})]},
    "Frame Material & look": {"fields": ["Shopify Gestell", "Massivholz", "Shopify Material", "variant", "name"],
        "table": [("Metal", r"stahl|steel|metall|metal|chrom|nickel|aluminium|\balu\b|eisen|messing|brass|edelstahl|stainless|kupfer"),
                  ("Wood", r"holz|wood|eich|oak|nuss|walnut|buche|beech|esche|\bash\b|teak|akazie|acacia|birke|birch|ahorn|maple|kirsch|mahagoni|bambus"),
                  ("Plastic", r"kunststoff|plastic|polypropylen|polyethylen|resin|fiberglas|fiberglass|acryl|schalensitz"),
                  ("Other", r"rattan|korb|geflecht|cane|wicker|rohr\b|bast|seil|rope"),
                  ("Textile", r"vollpolster|fully upholstered|ohne gestell")]},
    "Material & look": {"fields": ["Shopify Sitzbezug", "Shopify Gestell", "Massivholz", "Shopify Material", "variant", "name"], "first_match_wins": True, "reliability_factor": 0.7,
        "table": [("Leather", r"leder|leather|ecoleder|kunstleder|nubuk|cuero|nappa"),
                  ("Metal/outdoor frame", r"outdoor|stahl|steel|metall|aluminium|chrom|edelstahl|gartensessel"),
                  ("Exposed wood frame", r"holzgestell|massivholz|eich|oak|nuss|walnut|teak|holz|buche|esche|akazie|birke"),
                  ("Colour or pattern fabric", _COLOURFUL, {"rel": 0.7}),
                  ("Neutral fabric (grey, beige, white)", _NEUTRAL, {"rel": 0.7})]},
    "Frame Functionality": {"fields": ["name", "variant", "Shopify Ausstattung", "Printkatalogartikel Text"],
        "table": [("Swivel base", r"drehbar|drehsessel|swivel|rotat|drehgestell|drehfuss"),
                  ("Rocking/gliding base", r"schaukel|rocking|rocker|gleit|glider|wippe|schwing"),
                  ("Adjustable recline", r"relax|recliner|verstellbar|neigbar|liegefunktion|reclin|kippfunktion|funktionssessel|adjustable")]},
    "Add ons": {"fields": ["name", "variant", "Shopify Ausstattung", "Printkatalogartikel Text"], "binary": True,
        "table": [("Ottoman", r"inkl\.?\s*(fuss)?hocker|mit\s+(fuss)?hocker|mit\s+ottoman|with\s+ottoman|\+\s*hocker|incl\.?\s*ottoman|mit\s+fussteil|and\s+ottoman|und\s+hocker|\bottoman\b"),
                  ("Headrest", r"kopfstuetze|kopfteil|headrest|nackenstuetze|nackenkissen"),
                  (None, r"ohne\s+armlehne\w*|armless|without\s+armrest\w*"),
                  ("Armrest", r"armlehne|armrest|mit\s+armlehnen")]},
  },
  "shelving": {
    "Material & look": {"fields": ["Shopify Material", "material", "Shopify Farbe", "variant", "name"], "mixed_value": "Mix",
                        "mixed_fields": ["Shopify Material", "material", "Shopify Farbe", "variant"],
        "table": [("Dark wood", r"walnuss|nussbaum|walnut|wenge|(eich|oak|esche)\w*[\s,/-]*(dunkel|schwarz|black|dark)|dark oak|raeuchereiche"),
                  ("Light wood", r"eich|oak|esche|birke|birch|buche|beech|kiefer|pine|ahorn|maple|bambus|bamboo|furnier|holz|wood|sperrholz|plywood|multiplex"),
                  ("Black/metal frame", r"stahl|metall|steel|metal|draht|wire|aluminium|chrom|pulverbeschichtet|eisen|iron"),
                  ("White/light lacquer", r"lack|lacquer|weiss|white|cream|creme|offwhite|hellgrau|light gr[ae]y|beige|sand\b"),
                  ("Colourful or plastic", r"kunststoff|plastic|bunt|rot\b|red\b|blau|blue|gruen|green|gelb|yellow|orange|pink|rosa|petrol|lila|tuerkis|mint|multicolou?r")]},
    "Functionality": {"fields": ["name", "variant", "Shopify Ausstattung", "Shopify Größe", "Printkatalogartikel Text"],
        "table": [(None, r"ohne\s+tuer\w*(/-?\s*schublad\w*)?|ohne\s+schublad\w*|without\s+(doors|drawers)"),
                  ("Ladder/leaning", r"leiter|ladder|lehnregal|leaning|anlehn"),
                  ("Wall-mounted/floating", r"wandregal|wandmontage|wall[- ]mounted|floating|schwebe|haengeregal|wandboard|wandkonsole|konsole|wall shelf|wandmodul|wandbefestigung"),
                  ("Enclosed with doors or drawers", r"tuer|door|schublade|drawer|schrank|cabinet|sideboard|vitrine|klappe")]},
  },
  "portable_lamp": {
    "Material & look": {"fields": ["material", "Shopify Material", "variant", "name", "Shopify Gestell"],
        "table": [("Fabric shade", r"stoff|textil|fabric|leinen|linen|baumwolle|cotton|filz|felt"),
                  ("Ceramic/porcelain", r"keramik|ceramic|porzellan|porcelain|steinzeug"),
                  ("Glass", r"glas(?!ur)|glass|kristall"),
                  ("Plastic/silicone", r"kunststoff|plastic|silikon|silicone|polycarbonat|acryl|resin|polyethylen|gummi|\babs\b|\bpp\b"),
                  ("Metal", r"metall|metal|aluminium|\balu\b|stahl|steel|messing|brass|kupfer|edelstahl|eisen|bronze|zink|chrom")]},
    "Light control": {"fields": ["name", "variant", "Shopify Ausstattung", "Shopify Leuchtmittel", "Shopify Gestell", "Printkatalogartikel Text",
                                 "Printkatalogartikel Verkürzte Bezeichnung", "Printkatalogartikel Varianten"],
        "table": [("Colour-changing (RGB)", r"\brgb|farbwechsel|colou?r[- ]changing|multicolou?r led"),
                  ("Smart (app or remote)", r"\bapp\b|smart|bluetooth|wlan|wifi|fernbedienung|remote|alexa|homekit|zigbee"),
                  ("Tunable white (warm to cool)", r"tunable|farbtemperatur|warmweiss\s*(bis|to)\s*kaltweiss|2700.{0,6}6500|\bcct\b"),
                  ("Dimmable", r"dimm|touch[- ]?dimmer|stufenlos|dimmable"),
                  ("Fixed warm white", r"warmweiss|warm white|warmes weiss|\b200\s*k\b|\b3000\s*k\b",
                       {"unless": ["Dimmable", "Colour-changing (RGB)", "Smart (app or remote)", "Tunable white (warm to cool)"]})]},
    "Environment": {"fields": ["name", "variant", "main_category", "Shopify Ausstattung", "Printkatalogartikel Text", "Shopify Größe"],
        "table": [("Full outdoor (weather-rated)", r"\bip\s?x??\b|wetterfest|weatherproof|wasserdicht|waterproof|ganzjaehrig|allwetter|salzwasser|spritzwasser"),
                  ("Covered outdoor (balcony, terrace)", r"outdoor|aussen|balkon|terrasse|garten|camping|solar"),
                  ("Indoor only", r"\bindoor\b|innenbereich", {"unless": ["Covered outdoor (balcony, terrace)", "Full outdoor (weather-rated)"]})]},
  },
  "cup": {
    "Material": {"fields": ["material", "Shopify Material", "variant", "name"],
        "table": [("Glass", r"glas(?!ur)|glass|borosilikat"),
                  ("Enamel/metal", r"email|emaille|enamel|metall|edelstahl|stainless|\bstahl|aluminium|blech"),
                  ("Fine porcelain", r"porzellan|porcelain|bone china|fine china|knochenporzellan"),
                  ("Stoneware", r"steinzeug|stoneware|steingut|earthenware"),
                  ("Stoneware", r"keramik|ceramic", {"rel": 0.7}),
                  ("Other", r"kunststoff|plastic|melamin|bambus|bamboo|silikon|silicone|holz|wood|kork|papier|biomaterial|\bpla\b")]},
  },
}
FIELD_RELIABILITY_TYPE = {"name": "product_text", "variant": "product_text"}   # all other fields count as structured product attributes
TEXT_FIELDS_ALWAYS = ["name", "variant", "color_name", "material", "main_category", "sub_category"]

# ---- Absence defaults: value assumed when the text gives no sign of the alternatives -------------
# (value, reliability).  reliability 0.55 -> Medium, 0.35-0.45 -> Low.  Delete a line => 'Not determinable'.
ABSENCE_DEFAULTS = {
  "dining_table":  {"Functionality": ("Fixed & simple", 0.55)},
  "lounge_chair":  {"Frame Functionality": ("Fixed frame (no mechanism)", 0.55)},
  "shelving":      {"Functionality": ("Freestanding single unit", 0.40)},
  "portable_lamp": {"Environment": ("Indoor only", 0.35)},
}

# ---- Colours --------------------------------------------------------------------------------
COLOUR_GROUPS = ["Light Neutrals", "Warm Earth & Natural Woods", "Cool Grays & Minerals", "Deep Shadows & Graphites",
                 "Chromatic Accents & Jewel Tones"]
# (regex, group, flags) – applied in this order; matched text is consumed so 'hellgrau' is not also 'grau'.
COLOUR_LEXICON = [
  (r"bunt|mehrfarbig|multi\s?colou?r|\bmulti\b|colou?rful|rainbow|regenbogen|gemustert|patterned|gestreift|striped|\bkaro\b|floral", "Mix", ()),
  (r"transparent|glasklar|durchsichtig|crystal clear|\bklar\b|\bclear\b", None, ("transparent",)),
  (r"rostfrei", "Cool Grays & Minerals", ("metallic",)),
  (r"blaugrau|graublau|stahlblau|blue\s*gr[ae]y|gr[ae]y\s*blue|graugruen|gr[ae]y\s*green|salbeigrau", "Cool Grays & Minerals", ()),
  (r"dunkel\s?gr[ae]u|dunkel\s?gr[ae]y|dark\s?gr[ae]y|anthrazit|anthracite|graphit|graphite|schwarz|black|\bnero\b|kohle|charcoal|onyx|ebenholz|ebony|\bnoir\b|midnight|espresso", "Deep Shadows & Graphites", ()),
  (r"hell\s?gr[ae]u|hell\s?gr[ae]y|light\s?gr[ae]y|weiss|white|creme|cream|elfenbein|ivory|off[- ]?white|ecru|leinen|linen|alabaster|knochen|\bbone\b|milch|perlmutt|pearl|schnee|snow|chalk|kreide|vanille|vanilla|champagne|beige|\bsand|taupe|greige|\bnude\b|porzellan", "Light Neutrals", ()),
  (r"kupfer|copper|messing|brass|bronze|\bgold\b|vergoldet", "Warm Earth & Natural Woods", ("metallic",)),
  (r"silber|silver|chrom|nickel|aluminium|\balu\b|edelstahl|stainless|\bstahl|\bsteel|zinn|pewter|titan", "Cool Grays & Minerals", ("metallic",)),
  (r"braun|brown|eich|oak|walnuss|nussbaum|walnut|teak|buche|beech|birke|birch|esche|kirsch|cherry|ahorn|maple|akazie|acacia|holz|wood|bambus|bamboo|rattan|kork|cork|terrakotta|terracotta|rost|\brust|ocker|ochre|camel|cognac|karamel|caramel|zimt|cinnamon|honig|honey|lehm|\bclay\b|erde|earth|\bnatur|\bnature|tabak|tobacco|schoko|chocolate|mokka|mocha|kaffee|coffee|nougat|lebkuchen|hazel|haselnuss|nuss|mahagoni|mahogany", "Warm Earth & Natural Woods", ()),
  (r"gr[ae]u|gr[ae]y|stein|stone|beton|concrete|zement|cement|schiefer|slate|nebel|\bfog\b|\bmist\b|rauch|smoke|marmor|marble|mineral|taube|dove|kiesel|pebble|granit", "Cool Grays & Minerals", ()),
  (r"\brot\b|\bred\b|rot(?=braun|orange)|blau|blue|gruen|green|gelb|yellow|orange|pink|rosa|\brose\b|violett|violet|lila|purple|tuerkis|turquoise|petrol|smaragd|emerald|saphir|sapphire|rubin|ruby|bordeaux|burgund|aubergine|magenta|\bmint|oliv|cobalt|kobalt|ultramarin|flieder|lilac|lavendel|lavender|senf|mustard|koralle|coral|himmel|\bsky\b|navy|marine|indigo|fuchsia|beere|berry|tomate|tomato|\blime\b|apricot|aprikose|peach|pfirsich|blush|salbei|\bsage\b|eukalyptus|eucalyptus|jade|\bmoos|\bmoss|forest|tanne|lachs|salmon|zitrone|lemon|aqua|cyan|\bteal\b|plum|pflaume|dusty|candy|bonbon|bubblegum", "Chromatic Accents & Jewel Tones", ()),
]
COLOUR_TEXT_FIELDS = ["Shopify Farbe", "color_name", "variant", "name_tail"]   # first field that yields colour words is used

# colour group (or 'transparent'/'metallic') -> {allowed value: probability}. Attributes whose allowed values ARE the
# group names (e.g. 'Colour Group') use the identity mapping automatically.
COLOUR_VALUE_MAP = {
  "vase": {"Colour & finish": {
      "Light Neutrals": {"White/neutral": 1.0}, "Cool Grays & Minerals": {"White/neutral": 0.6, "Black/dark": 0.2},
      "Deep Shadows & Graphites": {"Black/dark": 1.0}, "Warm Earth & Natural Woods": {"White/neutral": 0.4, "Colourful or printed": 0.4},
      "Chromatic Accents & Jewel Tones": {"Colourful or printed": 1.0}, "Mix": {"Colourful or printed": 1.0},
      "transparent": {"Transparent/clear": 1.0}, "metallic": {"Metallic finish": 1.0}}},
  "shelving": {"Material & look": {
      "Light Neutrals": {"White/light lacquer": 0.6, "Light wood": 0.2}, "Warm Earth & Natural Woods": {"Light wood": 0.45, "Dark wood": 0.25},
      "Cool Grays & Minerals": {"Black/metal frame": 0.35, "White/light lacquer": 0.25}, "Deep Shadows & Graphites": {"Black/metal frame": 0.7, "Dark wood": 0.2},
      "Chromatic Accents & Jewel Tones": {"Colourful or plastic": 0.8}, "Mix": {"Mix": 0.7}}},
}
COLOUR_IMAGE_SETTINGS = {"dominant_min_share": 0.60, "second_min_share": 0.25, "kmeans_k": 4, "max_pixels": 6000, "drop_cluster_below": 0.05}

# ---- Dimension rules (cm) ---------------------------------------------------------------------
# thresholds are 'up to and including'; labels are matched to allowed values by their prefix (XS/S/M/L)
SEAT_RULES = {
  "round":       [(75, "XS"), (115, "S"), (145, "M"), (9999, "L")],      # diameter
  "square":      [(80, "XS"), (110, "S"), (140, "M"), (9999, "L")],      # side length
  "rectangular": [(110, "XS"), (150, "S"), (190, "M"), (9999, "L")],     # longer side
  "softness": 0.06,                                                       # blur of the thresholds (relative)
}
LOUNGE_SEAT_WIDTH_EDGES = [(65, "XS"), (75, "S"), (85, "M"), (9999, "L")]
LOUNGE_SEAT_WIDTH_ATTRIBUTES = ["Sitzbreite"]                            # ItemAttribute holding the seat width (cm)
LOUNGE_FALLBACK_OFFSET_CM = 13.0                                         # used if calibration data are insufficient
VASE_RULES = {"floor_height_cm": 55, "slender_ratio": 2.6, "slender_max_diameter_cm": 13, "tube_ratio_range": (1.4, 3.5),
              "tube_min_diameter_cm": 10, "round_max_ratio": 1.6}
TABLE_DIMENSION_ATTRIBUTES = {"length": "Grundlänge", "width": "Grundbreite", "diameter": "Grund-Durchmesser"}   # ItemAttribute names (cm)

# ---- YOUR NEW COLUMNS -> ATTRIBUTES -------------------------------------------------------------
# Example: you add a column 'tier' to the product file and an attribute 'Price tier' (values Entry; Mid; Premium) to the
# allowed-values file. Un-comment and adapt – no other code change is needed.
ATTRIBUTE_FROM_COLUMN = [
  # {"category": "*", "attribute": "Price tier", "column": "tier", "mode": "exact"},                      # value text equals an allowed value
  # {"category": "*", "attribute": "Price tier", "column": "tier", "mode": "map", "map": {"A": "Premium", "B": "Mid", "C": "Entry"}},
  # {"category": "*", "attribute": "Price segment", "column": "price", "mode": "bins",
  #  "edges": [50, 150], "labels": ["Entry", "Mid", "Premium"]},                                         # <=50, <=150, above
]

# %% [markdown]
# ## 2. Loading and validating the source files
# * File names and column names are found *safely* (case / spaces / separators are ignored, several encodings and delimiters are tried).
# * The discovered schema and the data-quality findings are displayed and written to `RES/data_quality_report.csv`.
# * Placeholders (`0`, `-`, `'-`, empty) are turned into real missing values **only where they mean "missing"** (text and measurements, not item-attribute values).

# %%
# ---------- logging helpers (everything is also written to RES/processing_log.txt at the end) ----------
LOG, DQ = [], []

def log(msg, level="INFO"):
    line = f"{datetime.now():%H:%M:%S} [{level}] {msg}"
    LOG.append(line); print(line)

def dq(area, check, count, severity="info", detail=""):
    """Record a data-quality finding. severity: info | warning | error"""
    DQ.append({"area": area, "check": check, "count": count, "severity": severity, "detail": detail})

# ---------- text helpers ----------
PLACEHOLDERS = {"", "0", "-", "'-", "–", "nan", "none", "null", "n/a", "na", "#n/a", "#nv", "unknown"}

def norm_key(s):
    """Letters+digits only, lower case, accents removed -> tolerant matching of names."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def fold_text(s):
    """Lower case + German transliteration + accents removed. Used for keyword matching (patterns are folded the same way)."""
    if s is None or (isinstance(s, float) and np.isnan(s)): return ""
    s = str(s).lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()

def clean_text(x, zero_is_missing=True):
    """Trim, collapse spaces, placeholder -> NaN (meaning is not changed)."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    s = re.sub(r"\s+", " ", str(x).replace("\u00a0", " ")).strip()
    low = s.lower()
    if low in PLACEHOLDERS and not (low == "0" and not zero_is_missing): return np.nan
    return s

def parse_number(x):
    """'1.234,56' | '1,5' | '3.403.382' | '(106)' (=negative) | '31.5' -> float; unparsable -> NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    s = str(x).strip().replace("\u00a0", "").replace(" ", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if s in ("", "-"): return np.nan
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    elif s.count(".") > 1 or re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try: v = float(s)
    except ValueError: return np.nan
    return -v if neg else v

def clean_id(x):
    s = clean_text(x, zero_is_missing=False)
    if isinstance(s, float): return s
    return re.sub(r"\.0+$", "", s)

# ---------- file discovery & reading ----------
TABLE_EXTS = [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"]

def find_file_by_spec(spec, label):
    pats = [norm_key(p) for p in spec["patterns"]]
    exts = spec.get("extensions", TABLE_EXTS)
    hits = []
    for p in SRC_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            k = norm_key(p.stem)
            for i, pat in enumerate(pats):
                if pat in k: hits.append((i, -p.stat().st_mtime, p)); break
    if not hits:
        msg = f"No file found for '{label}' (patterns {spec['patterns']}) in {SRC_DIR}"
        if spec.get("required"): raise FileNotFoundError(msg)
        log(msg, "WARN"); dq("files", f"missing optional file: {label}", 0, "warning", msg); return None
    hits.sort(key=lambda t: (t[0], t[1]))
    if len(hits) > 1:
        log(f"{label}: {len(hits)} candidate files, using {hits[0][2].name}", "WARN")
        dq("files", f"several candidate files for {label}", len(hits), "warning", "; ".join(h[2].name for h in hits))
    return hits[0][2]

def find_file(kind):
    return find_file_by_spec(CFG["files"][kind], kind)

def read_table(path):
    """Read csv/tsv/txt/xlsx/parquet as *strings*; try several encodings and delimiters. Returns (df, meta)."""
    path = Path(path); ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str, keep_default_na=False); return df, {"encoding": "excel", "delimiter": ""}
    if ext == ".parquet":
        return pd.read_parquet(path).astype(str), {"encoding": "parquet", "delimiter": ""}
    raw = path.read_bytes()
    enc_used, text = None, None
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try: text = raw.decode(enc); enc_used = enc; break
        except UnicodeDecodeError: continue
    first = next((l for l in text.splitlines() if l.strip()), "")
    first_unquoted = re.sub(r'"[^"]*"', "", first)
    delim = max([",", ";", "\t", "|"], key=lambda d: first_unquoted.count(d))
    df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False, na_filter=False)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df, {"encoding": enc_used, "delimiter": repr(delim)}

def resolve_columns(df, mapping, table, required=()):
    """Map logical names to real headers (case/space/separator-insensitive). Returns {logical: header or None}."""
    keys = {}
    for c in df.columns: keys.setdefault(norm_key(c), c)
    found = {}
    for logical, cands in mapping.items():
        found[logical] = next((keys[norm_key(c)] for c in cands if norm_key(c) in keys), None)
    miss_req = [k for k in required if not found.get(k)]
    if miss_req:
        raise KeyError(f"{table}: required column(s) {miss_req} not found. Headers are: {list(df.columns)}. Edit CFG['columns']['{table}'].")
    for k, v in found.items():
        if v is None: dq(table, f"column not found: {k}", 0, "warning", f"candidates {mapping[k]}")
    return found

SCHEMA_ROWS = []
def record_schema(kind, path, df, meta, found):
    SCHEMA_ROWS.append({"source": kind, "file": path.name, "rows": len(df), "columns": len(df.columns),
                        "encoding": meta["encoding"], "delimiter": meta["delimiter"],
                        "mapped": ", ".join(f"{k}->{v}" for k, v in found.items() if v) if found else "",
                        "unmapped_required_or_optional": ", ".join(k for k, v in found.items() if not v) if found else ""})

# %%
# ---------- 2.1 locate and read everything ----------
SOURCES, FOUND = {}, {}
for kind in ["products", "item_attributes", "media", "allowed_values", "sales"]:
    p = find_file(kind)
    if p is None: continue
    df, meta = read_table(p)
    SOURCES[kind] = df
    if kind in CFG["columns"]:
        req = {"products": ["product_id", "product_name", "sub_category"], "item_attributes": ["product_id", "name", "value"],
               "media": ["product_id", "url"], "allowed_values": ["category", "attribute", "value"]}[kind]
        FOUND[kind] = resolve_columns(df, CFG["columns"][kind], kind, req)
    elif kind == "sales":
        FOUND[kind] = resolve_columns(df, CFG["sales"]["columns"], "sales", ["product_id"])
    else:
        FOUND[kind] = {}
    record_schema(kind, p, df, meta, FOUND[kind])
    log(f"read {kind}: {p.name} -> {len(df):,} rows x {len(df.columns)} columns (encoding {meta['encoding']}, delimiter {meta['delimiter']})")

PDF_PATH = find_file("pdf")
if PDF_PATH is not None:
    SCHEMA_ROWS.append({"source": "pdf", "file": PDF_PATH.name, "rows": "", "columns": "", "encoding": "", "delimiter": "", "mapped": "", "unmapped_required_or_optional": ""})
display(pd.DataFrame(SCHEMA_ROWS))

# %%
# ---------- 2.2 normalise the product table ----------
def load_products():
    raw, m = SOURCES["products"].copy(), FOUND["products"]
    out = pd.DataFrame(index=raw.index)
    for logical, header in m.items():
        if header is not None: out[logical] = raw[header]
        else: out[logical] = np.nan
    used = {h for h in m.values() if h}

    # your extra columns (price, tier ...) -> logical names, typed
    extra_found = {}
    for logical, spec in CFG["extra_product_fields"].items():
        keys = {norm_key(c): c for c in raw.columns if c not in used}
        header = next((keys[norm_key(c)] for c in spec["candidates"] if norm_key(c) in keys), None)
        if header is None:
            log(f"extra field '{logical}' not in product file (candidates {spec['candidates'][:4]}...) - skipped, add the column when available"); continue
        used.add(header); extra_found[logical] = header
        out[logical] = raw[header].map(parse_number) if spec["type"] == "number" else raw[header].map(clean_text)
    # any other column is carried through unchanged (auto-typed)
    carried = []
    if CFG["carry_unmapped_product_columns"]:
        for c in raw.columns:
            if c in used: continue
            name = c if c not in out.columns else f"src_{c}"
            col = raw[c].map(clean_text)
            num = col.map(parse_number)
            out[name] = num if (col.notna().sum() > 0 and num.notna().sum() >= 0.95 * col.notna().sum()) else col
            carried.append(name)
    log(f"products: extra fields found = {extra_found or 'none'}; other carried columns = {carried or 'none'}")

    out["product_id"] = out["product_id"].map(clean_id)
    for c in ["product_name", "manufacturer_id", "top_category", "main_category", "sub_category", "variant", "color_name", "material"]:
        out[c] = out[c].map(clean_text)
    div = CFG["units"]["dimension_divisor_to_cm"]
    for c in ["length", "depth", "height"]:
        out[c + "_cm"] = pd.to_numeric(out[c].map(parse_number), errors="coerce").replace(0, np.nan) / div
    out["weight_g"] = pd.to_numeric(out["weight"].map(parse_number), errors="coerce").replace(0, np.nan)
    out = out.drop(columns=["length", "depth", "height", "weight"])
    out["sub_category_key"] = out["sub_category"].map(lambda s: norm_key(s) if isinstance(s, str) else "")

    n0 = len(out)
    out = out[out["product_id"].notna()]
    dq("products", "rows without product id (removed)", n0 - len(out), "warning" if n0 - len(out) else "info")
    # duplicates: keep the most complete row
    dups = out["product_id"].duplicated(keep=False)
    dq("products", "rows sharing a product id", int(dups.sum()), "warning" if dups.any() else "info",
       "kept the most complete row per product id; examples: " + ", ".join(out.loc[dups, "product_id"].unique()[:5]))
    out = (out.assign(_filled=out.notna().sum(axis=1)).sort_values("_filled", ascending=False, kind="stable")
              .drop_duplicates("product_id").drop(columns="_filled").sort_index())
    # placeholders per column (information only)
    ph = {c: int(SOURCES["products"][h].map(lambda v: str(v).strip().lower() in PLACEHOLDERS).sum()) for c, h in m.items() if h}
    dq("products", "placeholder values (0/-/empty) turned into missing, by column", sum(ph.values()), "info", json.dumps(ph))
    return out.reset_index(drop=True), extra_found

PRODUCTS_ALL, EXTRA_FOUND = load_products()
log(f"products normalised: {len(PRODUCTS_ALL):,} unique product ids")

# ---------- 2.3 item attributes (long) ----------
def load_item_attributes():
    if "item_attributes" not in SOURCES: return pd.DataFrame(columns=["product_id", "attr_set", "attr_name", "attr_value"])
    raw, m = SOURCES["item_attributes"], FOUND["item_attributes"]
    ia = pd.DataFrame({"product_id": raw[m["product_id"]].map(clean_id),
                       "attr_set": raw[m["set"]].map(clean_text) if m.get("set") else np.nan,
                       "attr_name": raw[m["name"]].map(clean_text),
                       "attr_value": raw[m["value"]].map(lambda v: clean_text(v, zero_is_missing=False))})
    n0 = len(ia)
    ia = ia.dropna(subset=["product_id", "attr_name", "attr_value"])
    dq("item_attributes", "rows without id / name / value (removed)", n0 - len(ia), "info")
    exact = ia.duplicated()
    dq("item_attributes", "exact duplicate rows (removed)", int(exact.sum()), "warning" if exact.any() else "info")
    ia = ia[~exact]
    conflicting = ia.duplicated(["product_id", "attr_name"], keep=False)
    dq("item_attributes", "same product+attribute with different values (all values kept, joined by ' | ')",
       int(ia[conflicting][["product_id", "attr_name"]].drop_duplicates().shape[0]), "warning" if conflicting.any() else "info")
    return ia.reset_index(drop=True)

ITEM_ATTR = load_item_attributes()
log(f"item attributes: {len(ITEM_ATTR):,} clean rows for {ITEM_ATTR['product_id'].nunique():,} products")

# ---------- 2.4 media ----------
def load_media():
    if "media" not in SOURCES: return pd.DataFrame(columns=["product_id", "media_id", "order", "url"])
    raw, m = SOURCES["media"], FOUND["media"]
    md = pd.DataFrame({"product_id": raw[m["product_id"]].map(clean_id),
                       "media_id": raw[m["media_id"]].map(clean_id) if m.get("media_id") else np.nan,
                       "order": pd.to_numeric(raw[m["order"]].map(parse_number), errors="coerce") if m.get("order") else np.nan,
                       "url": raw[m["url"]].map(clean_text)})
    n0 = len(md)
    md = md.dropna(subset=["product_id", "url"])
    dq("media", "rows without product id / url (removed)", n0 - len(md), "info")
    pre = CFG["images"]["url_prefix"]
    md["url"] = md["url"].map(lambda u: u if re.match(r"^(https?|file)://", u) or Path(u).exists() else pre + u)
    bad = ~md["url"].str.match(r"^(https?|file)://") & ~md["url"].map(lambda u: Path(u).exists())
    dq("media", "links without http(s) scheme (kept; need images.url_prefix or local file)", int(bad.sum()), "warning" if bad.any() else "info")
    dup = md.duplicated(["product_id", "url"])
    dq("media", "duplicate product+url rows (removed)", int(dup.sum()), "info")
    md = md[~dup]
    sentinel = md["order"] > 100
    dq("media", "rows with sentinel/odd 'Order' value (>100, ranked last)", int(sentinel.sum()), "info", "e.g. 999999")
    return md.reset_index(drop=True)

MEDIA = load_media()
log(f"media: {len(MEDIA):,} clean image links for {MEDIA['product_id'].nunique():,} products")

# %%
# ---------- 2.5 data-quality summary so far ----------
display(pd.DataFrame(DQ))

# %% [markdown]
# ## 3. Extract and validate allowed attributes and values
# Only a value listed for that category+attribute can be exported. If there is no evidence, the pipeline writes the explicit
# `Not determinable` sentinel; it is clearly tagged as a sentinel, not passed off as an allowed value.
# Matrix categories are matched by name normalisation/containment; if a mapping is ambiguous, edit `CFG['targets'][...]['matrix_category']` in Section 1.

# %%
# ---------- allowed values --------------------------------------------------------------------
def load_allowed_values():
    raw, m = SOURCES["allowed_values"], FOUND["allowed_values"]
    av = pd.DataFrame({"category": raw[m["category"]].map(clean_text), "attribute": raw[m["attribute"]].map(clean_text),
                       "value": raw[m["value"]].map(lambda v: clean_text(v, zero_is_missing=False))}).dropna()
    dup = av.duplicated()
    dq("allowed_values", "duplicate category/attribute/value rows (removed)", int(dup.sum()), "warning" if dup.any() else "info")
    av = av[~dup].reset_index(drop=True)
    allowed = {}
    for (cat, attr), g in av.groupby(["category", "attribute"], sort=False):
        allowed.setdefault(cat, {})[attr] = list(g["value"])
    return av, allowed

ALLOWED_DF, ALLOWED = load_allowed_values()
display(ALLOWED_DF.groupby(["category", "attribute"], sort=False)["value"]
        .agg(n_values="count", values=lambda s: "; ".join(s)).reset_index())

# %% [markdown]
# ## 4. Resolve category keys and select the target sub-categories
# Product category values, attribute-side category values and their counts are reported separately; conflicting category
# labels do not silently reassign a product. The optional `include_products_by_item_attribute_category` flag can widen selection.

# %%
TARGET_DWH_VALUES = {}
MATRIX_CATEGORY = {}
CATEGORY_MAP_ROWS = []

def resolve_matrix_category(target_name, target_spec):
    explicit = target_spec.get("matrix_category")
    categories = list(ALLOWED.keys())
    if explicit:
        if explicit not in ALLOWED: raise KeyError(f"matrix category {explicit!r} for {target_name} is not in allowed values")
        return explicit
    kn = norm_key(target_name)
    # Prefer exact normalized equality, then a single word-boundary-ish containment match.
    exact = [c for c in categories if norm_key(c) == kn]
    if len(exact) == 1: return exact[0]
    cand = [c for c in categories if kn in norm_key(c) or norm_key(c) in kn]
    if len(cand) == 1: return cand[0]
    if target_name == "Esstisch" and "Design-Esstisch" in categories: return "Design-Esstisch"
    raise KeyError(f"Could not resolve allowed-value category for {target_name!r}. Candidates={cand}; set CFG['targets'][{target_name!r}]['matrix_category'].")

for target, spec in CFG["targets"].items():
    MATRIX_CATEGORY[target] = resolve_matrix_category(target, spec)
    vals = spec.get("dwh_values") or [target]
    if isinstance(vals, str): vals = [vals]
    TARGET_DWH_VALUES[target] = vals
    for dwh_cat in vals:
        pc = PRODUCTS_ALL[PRODUCTS_ALL["sub_category_key"] == norm_key(dwh_cat)]
        ia_cat = ITEM_ATTR[(ITEM_ATTR["attr_name"].map(norm_key) == norm_key("Shopify Sub Category")) &
                           (ITEM_ATTR["attr_value"].map(norm_key) == norm_key(dwh_cat))]
        CATEGORY_MAP_ROWS.append({"target": target, "dwh_category_value": dwh_cat, "matrix_category": MATRIX_CATEGORY[target],
                                  "product_file_count": len(pc), "item_attribute_category_count": ia_cat["product_id"].nunique(),
                                  "allowed_attribute_count": len(ALLOWED[MATRIX_CATEGORY[target]])})

# Build selected product rows from the canonical DWH source data.
sel_parts = []
for target, spec in CFG["targets"].items():
    keys = {norm_key(v) for v in TARGET_DWH_VALUES[target]}
    mask = PRODUCTS_ALL["sub_category_key"].isin(keys)
    if CFG["include_products_by_item_attribute_category"]:
        ia_cats = ITEM_ATTR[(ITEM_ATTR["attr_name"].map(norm_key) == norm_key("Shopify Sub Category")) &
                            (ITEM_ATTR["attr_value"].map(lambda x: norm_key(x) in keys))]
        mask = mask | PRODUCTS_ALL["product_id"].isin(ia_cats["product_id"])
    sub = PRODUCTS_ALL.loc[mask].copy()
    sub["subcategory"] = target
    sub["matrix_category"] = MATRIX_CATEGORY[target]
    sel_parts.append(sub)
SELECTED = pd.concat(sel_parts, ignore_index=True) if sel_parts else PRODUCTS_ALL.iloc[:0].copy()

# Detect duplicate IDs across target categories (ambiguous source data).
if SELECTED["product_id"].duplicated().any():
    dup_ids = SELECTED.loc[SELECTED["product_id"].duplicated(False), "product_id"].unique()
    dq("selection", "product ids selected into more than one target category", len(dup_ids), "error", ", ".join(dup_ids[:10]))
    SELECTED = SELECTED.drop_duplicates("product_id", keep="first")
else:
    dq("selection", "product ids selected into more than one target category", 0, "info")

CATEGORY_COUNTS = SELECTED.groupby("subcategory").size().reindex(CFG["targets"].keys(), fill_value=0).rename("products").reset_index()
display(pd.DataFrame(CATEGORY_MAP_ROWS))
display(CATEGORY_COUNTS)
log(f"selected {len(SELECTED):,} products from {SELECTED['subcategory'].nunique()} target sub-categories")

# Compare DWH and attribute-file category labels for the selected IDs.
_sub_attr = ITEM_ATTR[ITEM_ATTR["attr_name"].map(norm_key) == norm_key("Shopify Sub Category")].copy()
_sub_attr = _sub_attr[_sub_attr["product_id"].isin(SELECTED["product_id"])].groupby("product_id")["attr_value"].agg(lambda x: " | ".join(sorted(set(x.dropna())))).rename("attribute_sub_category")
CATEGORY_CHECK = SELECTED[["product_id", "subcategory", "sub_category"]].merge(_sub_attr, on="product_id", how="left")
CATEGORY_CHECK["category_conflict"] = CATEGORY_CHECK.apply(lambda r: pd.notna(r["attribute_sub_category"]) and bool(str(r["attribute_sub_category"]).strip()) and norm_key(r["sub_category"]) != norm_key(r["attribute_sub_category"]), axis=1)
_conflicts = CATEGORY_CHECK[CATEGORY_CHECK["category_conflict"]]
dq("selection", "selected products with conflicting DWH and ItemAttribute sub-category", len(_conflicts), "warning" if len(_conflicts) else "info")
if len(_conflicts): display(_conflicts.head(20))

# %% [markdown]
# ## 5. Join product data, existing attributes, image links and extra sources
# All joins are performed one-to-one / many-to-one after duplicate checks; the selected product row count is asserted.
# Existing attributes remain long-form here so assignments can be traced back to the original source and value.

# %%
# Add any extra one-row-per-product source files configured in CFG['extra_sources'].
def load_extra_sources(products):
    enriched = products.copy()
    for spec in CFG.get("extra_sources", []):
        path = find_file_by_spec({"patterns": spec["patterns"], "required": False}, spec["name"])
        if path is None: continue
        raw, meta = read_table(path)
        candidates = spec.get("key_candidates", ["ItemId"])
        key = resolve_columns(raw, {"product_id": candidates}, spec["name"], ["product_id"])["product_id"]
        keep = spec.get("keep_columns")
        keep = keep or [c for c in raw.columns if c != key]
        x = raw[[key] + [c for c in keep if c in raw.columns]].copy()
        x = x.rename(columns={key: "product_id"})
        x["product_id"] = x["product_id"].map(clean_id)
        for c in x.columns:
            if c != "product_id": x[c] = x[c].map(clean_text)
        # Numeric coercion is configured per source to avoid changing category meanings.
        for c in spec.get("numeric_columns", []):
            if c in x: x[c] = x[c].map(parse_number)
        ndup = int(x["product_id"].duplicated(keep=False).sum())
        if ndup:
            dq(spec["name"], "duplicate product keys in extra source (kept first row)", ndup, "warning")
            x = x.drop_duplicates("product_id", keep="first")
        prefix = spec.get("prefix", "")
        x = x.rename(columns={c: prefix + str(c) for c in x.columns if c != "product_id"})
        before = len(enriched)
        enriched = enriched.merge(x, on="product_id", how="left", validate="one_to_one")
        assert len(enriched) == before, f"Extra source {spec['name']} duplicated product rows"
        log(f"extra source {spec['name']}: {path.name}; joined {enriched.filter(regex='^'+re.escape(prefix)).notna().any(axis=1).sum():,} selected rows")
    return enriched

SELECTED = load_extra_sources(SELECTED)

# Pre-aggregate all ItemAttribute rows for selected IDs (one row per key and attribute after this step).
_SELECTED_IDS = set(SELECTED["product_id"].dropna())
SELECTED_ATTR = ITEM_ATTR[ITEM_ATTR["product_id"].isin(_SELECTED_IDS)].copy()
ATTR_MAP = defaultdict(dict)
for (pid, name), g in SELECTED_ATTR.groupby(["product_id", "attr_name"], sort=False):
    vals = list(dict.fromkeys(str(v).strip() for v in g["attr_value"].dropna() if str(v).strip()))
    if vals:
        ATTR_MAP[pid][name] = vals
        ATTR_MAP[pid][norm_key(name)] = vals

# Select best images: primary/clean product views first; lifestyle and detail URLs are kept as fallbacks.
def _image_rank(row):
    order = row["order"] if pd.notna(row["order"]) and row["order"] < 100 else 9999
    u = fold_text(row["url"])
    penalty = sum(1 for w in CFG["images"]["lifestyle_url_words"] if fold_text(w) in u)
    return (penalty, order, row["url"])

IMAGE_LINKS = MEDIA[MEDIA["product_id"].isin(_SELECTED_IDS)].copy()
IMAGE_LINKS["_rank"] = IMAGE_LINKS.apply(_image_rank, axis=1)
IMAGE_LINKS = IMAGE_LINKS.sort_values("_rank", kind="stable")
IMAGE_LINKS = IMAGE_LINKS.groupby("product_id", sort=False).head(CFG["images"]["max_images_per_product"]).drop(columns="_rank")
IMAGE_LINK_MAP = IMAGE_LINKS.groupby("product_id")["url"].agg(list).to_dict()
SELECTED["image_urls"] = SELECTED["product_id"].map(IMAGE_LINK_MAP).apply(lambda x: x if isinstance(x, list) else [])
_ATTR_COUNTS = SELECTED_ATTR.groupby("product_id")["attr_name"].nunique().to_dict()
SELECTED["existing_attribute_count"] = SELECTED["product_id"].map(lambda p: int(_ATTR_COUNTS.get(p, 0)))
SELECTED["image_link_count"] = SELECTED["product_id"].map(lambda p: len(IMAGE_LINK_MAP.get(p, [])))

# Explicitly report missing joins.
ids = set(SELECTED["product_id"])
missing_attr = SELECTED.loc[~SELECTED["product_id"].isin(set(SELECTED_ATTR["product_id"])), ["product_id", "product_name", "subcategory"]]
missing_img = SELECTED.loc[SELECTED["image_link_count"] == 0, ["product_id", "product_name", "subcategory"]]
dq("joins", "selected products with no ItemAttribute rows", len(missing_attr), "warning" if len(missing_attr) else "info")
dq("joins", "selected products with no image links", len(missing_img), "warning" if len(missing_img) else "info")
dq("joins", "selected product keys without source product row", int(SELECTED["product_id"].isna().sum()), "error")

# Product-attribute wide lookup helper used by the attribute engine.
def get_attr_values(product_id, attribute_names):
    for n in attribute_names:
        vals = ATTR_MAP.get(product_id, {}).get(n) or ATTR_MAP.get(product_id, {}).get(norm_key(n))
        if vals: return vals, n
    return [], None

def get_attr_text(product_id, attribute_names):
    vals, src = get_attr_values(product_id, attribute_names)
    return ("; ".join(vals), src) if vals else ("", None)

JOIN_SUMMARY = (SELECTED.groupby("subcategory", sort=False)
                .agg(products=("product_id", "nunique"), with_any_attributes=("existing_attribute_count", lambda x: int((x > 0).sum())),
                     with_images=("image_link_count", lambda x: int((x > 0).sum())))
                .reset_index())
display(JOIN_SUMMARY)
display(pd.concat([missing_attr.assign(issue="missing attributes"), missing_img.assign(issue="missing image links")]).head(30))
log(f"joined: {len(SELECTED):,} unique selected products; {SELECTED['image_link_count'].gt(0).sum():,} have images; {SELECTED['existing_attribute_count'].gt(0).sum():,} have attributes")

# %% [markdown]
# ## 6. Extract visual examples and style rules from `Designstyles.pdf`
# The PDF contains visual product boards, not written per-product labels. We preserve both:
# 1. extract each embedded product image and keep its page/category/style label for the few-shot image-similarity model;
# 2. record the specialist's visible style families as concrete interpretation rules (used for prompts and audit).
#
# **Interpretation used from the example boards:**
# | Label (allowed value) | Visual logic consistently shown in the PDF |
# |---|---|
# | Minimalistic | Plain, restrained forms; simple geometry; calm neutrals; little or no decoration. |
# | Fashionable | Current/playful statement forms, vivid colour, decorative/glazed/patterned finishes, trend-led shapes. |
# | Sculptural | Strong, unusual silhouette, expressive base, distinctive structure/curves or object-like form. |
# | Classic | Timeless/traditional proportions and materials; familiar forms; warm woods, traditional upholstery/details or classic cupware. |
#
# The PDF directly shows tables (pp. 1–4), lounge seating (5–8), cups (9–12), shelving (13–14), and lamps (15–18).
# It has **no vase examples**; style for vases is therefore not claimed as PDF-validated and is low-confidence unless other strong evidence exists.

# %%
# ---------- PDF extraction: can be skipped if unavailable or unreadable -----------------------
PDF_EXAMPLES = pd.DataFrame(columns=["example_id", "pdf_page", "pdf_xref", "rule_key", "style_value", "image_path", "width", "height"])
PDF_RULES = pd.DataFrame(columns=["rule_key", "page", "heading_text", "style_value", "interpretation"])
PDF_LIMITATION = ""

STYLE_INTERPRETATIONS = {
    "Minimalistic": "Restrained, simple geometry; calm neutral finishes; little visual ornament.",
    "Fashionable": "Trend-led/playful statement shape or bright colour/pattern/glaze.",
    "Sculptural": "Unusual object-like silhouette, strong curves, expressive/architectural base or structure.",
    "Classic": "Timeless/traditional proportions and materials; recognisable classic construction or detailing.",
}

try:
    import fitz  # PyMuPDF (optional; if missing, assignments continue from other evidence)
    from PIL import Image, ImageChops, ImageStat
    if PDF_PATH is None:
        PDF_LIMITATION = "Designstyles.pdf not found in SRC; no PDF examples used."
        log(PDF_LIMITATION, "WARN")
    else:
        pdf_doc = fitz.open(PDF_PATH)
        example_rows, rule_rows = [], []
        extracted_xrefs = set()
        for page_index, page in enumerate(pdf_doc):
            page_no = page_index + 1
            rule_key = next((rk for rk, (lo, hi) in CFG["pdf"]["page_ranges"].items() if lo <= page_no <= hi), None)
            heading = (page.get_text("text") or "").strip().splitlines()
            heading_text = heading[0].strip() if heading else ""
            heading_key = norm_key(heading_text)
            style = next((v for k, v in CFG["pdf"]["style_headings"].items() if norm_key(k) == heading_key), None)
            rule_rows.append({"rule_key": rule_key or "unmapped", "page": page_no, "heading_text": heading_text,
                              "style_value": style or "", "interpretation": STYLE_INTERPRETATIONS.get(style, "")})
            if rule_key is None: continue
            for info in page.get_image_info(xrefs=True):
                xref = int(info.get("xref", 0))
                if not xref: continue
                pix_w, pix_h = int(info.get("width", 0)), int(info.get("height", 0))
                if min(pix_w, pix_h) < CFG["pdf"]["min_example_side_px"]: continue
                try:
                    raw = pdf_doc.extract_image(xref)["image"]
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    # Remove only the near-white page background; keep product + soft shadow for shape evidence.
                    arr = np.asarray(im)
                    mask = np.any(arr < 248, axis=2)
                    ys, xs = np.where(mask)
                    if len(xs):
                        pad = max(2, int(0.03 * max(im.size)))
                        box = (max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
                               min(im.width, int(xs.max())+pad+1), min(im.height, int(ys.max())+pad+1))
                        if box[2] - box[0] >= 24 and box[3] - box[1] >= 24: im = im.crop(box)
                    target_px = 448
                    im.thumbnail((target_px, target_px), Image.Resampling.LANCZOS)
                    digest = hashlib.sha1(raw).hexdigest()[:10]
                    out_path = DIRS["pdf_examples"] / f"page{page_no:02d}_xref{xref}_{digest}.jpg"
                    if not out_path.exists(): im.save(out_path, quality=94, optimize=True)
                    example_id = f"p{page_no:02d}_x{xref}_{digest}"
                    example_rows.append({"example_id": example_id, "pdf_page": page_no, "pdf_xref": xref, "rule_key": rule_key,
                                         "style_value": style, "image_path": str(out_path), "width": im.width, "height": im.height})
                    extracted_xrefs.add(xref)
                except Exception as e:
                    dq("pdf", "embedded image extraction error", 1, "warning", f"page={page_no}; xref={xref}; {type(e).__name__}: {e}")
        PDF_EXAMPLES = pd.DataFrame(example_rows).drop_duplicates("example_id")
        PDF_RULES = pd.DataFrame(rule_rows)
        if PDF_EXAMPLES.empty:
            PDF_LIMITATION = "PDF was readable but had no extractable embedded product images in configured page ranges."
        PDF_EXAMPLES.to_csv(DIRS["pdf_examples"] / "pdf_example_manifest.csv", index=False, encoding="utf-8-sig")
        PDF_RULES.to_csv(DIRS["pdf_examples"] / "pdf_page_style_rules.csv", index=False, encoding="utf-8-sig")
        log(f"PDF: extracted {len(PDF_EXAMPLES):,} example images; {PDF_EXAMPLES['style_value'].nunique() if len(PDF_EXAMPLES) else 0} styles; pages {len(PDF_RULES)}")
        if PDF_LIMITATION: log(PDF_LIMITATION, "WARN")
except Exception as e:
    PDF_LIMITATION = f"PDF processing failed ({type(e).__name__}: {e}); continue without PDF visual examples."
    log(PDF_LIMITATION, "WARN")
    dq("pdf", "PDF unavailable or unreadable", 1, "warning", PDF_LIMITATION)

display(PDF_RULES.head(20))
display(PDF_EXAMPLES.groupby(["rule_key", "style_value"], dropna=False).size().rename("image_examples").reset_index() if len(PDF_EXAMPLES) else PDF_EXAMPLES)

# %% [markdown]
# ## 7. Download, cache and validate product images (best-effort)
# **No credentials are hard-coded.** Links are read from `ItemMedia.csv`; optional auth is loaded only from an environment variable.
# Safe connection/read timeouts, bounded retries, byte limits, image decoding, minimum resolution and persistent URL-hash caching
# prevent an individual broken link from stopping the workflow. The run can continue without images.
#
# Product images are ranked to prefer clean studio/packshot views over URLs containing lifestyle/detail words. A foreground
# crop is prepared from the best available image; Section 8 optionally improves it with a target-aware OWLv2 detector.

# %%
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None; REQUESTS_AVAILABLE = False
try:
    from PIL import Image, ImageOps, ImageFilter
    PIL_AVAILABLE = True
except Exception:
    Image = ImageOps = ImageFilter = None; PIL_AVAILABLE = False

IMAGE_STATUS_PATH = RES_DIR / "image_download_status.csv"
_prev_status = {}
if IMAGE_STATUS_PATH.exists():
    try:
        _p = pd.read_csv(IMAGE_STATUS_PATH, dtype=str, keep_default_na=False)
        if {"url_hash", "status", "cache_path"}.issubset(_p.columns):
            _prev_status = {r["url_hash"]: r for _, r in _p.iterrows()}
    except Exception: pass


def _safe_url(url):
    from urllib.parse import urlsplit
    p = urlsplit(str(url))
    return p.scheme.lower() in {"http", "https"} and bool(p.netloc) and not p.username and not p.password

def _url_for_download(url):
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
    prefix = CFG["images"].get("url_prefix", "")
    if not re.match(r"^https?://", str(url), re.I): url = prefix + str(url)
    if not CFG["images"].get("resize_query"): return url
    p = urlsplit(url); q = dict(parse_qsl(p.query)); q.update(dict(parse_qsl(CFG["images"]["resize_query"])))
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(q), p.fragment))

def _image_hash(url): return hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:24]

def _validate_and_save_image(content, out_path):
    if not PIL_AVAILABLE: return False, "Pillow not installed"
    if len(content) > CFG["images"]["max_bytes"]: return False, "exceeds max_bytes"
    try:
        im = Image.open(io.BytesIO(content))
        im.verify()
        im = Image.open(io.BytesIO(content))
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if min(w, h) < CFG["images"]["min_side_px"]: return False, f"image too small {w}x{h}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, "JPEG", quality=92, optimize=True)
        return True, f"{w}x{h}"
    except Exception as e:
        return False, f"invalid image: {type(e).__name__}: {e}"

def _download_one(task):
    pid, url, rank, order = task
    url2 = _url_for_download(url)
    uh = _image_hash(url2)
    raw_path = DIRS["cache_images_raw"] / f"{pid}_{uh}.jpg"
    lifestyle = any(fold_text(w) in fold_text(url) for w in CFG["images"]["lifestyle_url_words"])
    if raw_path.exists():
        try:
            im = Image.open(raw_path); im.verify(); im.close()
            return {"product_id": pid, "url": url, "url_hash": uh, "rank": rank, "order": order, "status": "cached",
                    "cache_path": str(raw_path), "width": None, "height": None, "lifestyle": lifestyle, "error": ""}
        except Exception:
            try: raw_path.unlink()
            except Exception: pass
    if not CFG["images"]["download"]:
        prev = _prev_status.get(uh, {})
        return {"product_id": pid, "url": url, "url_hash": uh, "rank": rank, "order": order, "status": "not_downloaded",
                "cache_path": "", "width": None, "height": None, "lifestyle": lifestyle, "error": "images.download=False"}
    if prev.get("status") == "failed" and not CFG["images"]["retry_failed_downloads"]:
        return {"product_id": pid, "url": url, "url_hash": uh, "rank": rank, "order": order, "status": "failed_cached",
                "cache_path": "", "width": None, "height": None, "lifestyle": lifestyle, "error": prev.get("error", "previous download failed")}
    if not REQUESTS_AVAILABLE:
        return {"product_id": pid, "url": url, "url_hash": uh, "rank": rank, "order": order, "status": "failed",
                "cache_path": "", "width": None, "height": None, "lifestyle": lifestyle, "error": "requests not installed"}
    # Optional local image path: only allow paths inside SRC to avoid reading arbitrary local files.
    local_candidate = None
    try:
        from urllib.parse import urlsplit, unquote
        if str(url2).lower().startswith("file://"):
            local_candidate = Path(unquote(urlsplit(str(url2)).path)).resolve()
        elif not re.match(r"^https?://", str(url2), re.I) and Path(str(url2)).exists():
            local_candidate = Path(str(url2)).resolve()
    except Exception: local_candidate = None
    if local_candidate is not None:
        try:
            local_candidate.relative_to(SRC_DIR.resolve())
            content = local_candidate.read_bytes()
            ok, msg = _validate_and_save_image(content, raw_path)
            if not ok: raise ValueError(msg)
            im = Image.open(raw_path); w, h = im.size; im.close()
            return {"product_id": pid, "url": url, "url_hash": uh, "rank": rank, "order": order, "status": "downloaded",
                    "cache_path": str(raw_path), "width": w, "height": h, "lifestyle": lifestyle, "error": ""}
        except Exception as e:
            return {"product_id": pid, "url": url, "url_hash": uh, "rank": rank, "order": order, "status": "failed",
                    "cache_path": "", "width": None, "height": None, "lifestyle": lifestyle, "error": f"local image rejected: {type(e).__name__}: {e}"[:500]}
    if not _safe_url(url2):
        return {"product_id": pid, "url": url, "url_hash": uh, "rank": rank, "order": order, "status": "failed",
                "cache_path": "", "width": None, "height": None, "lifestyle": lifestyle, "error": "unsafe or unsupported URL scheme/authority"}
    headers = {"User-Agent": CFG["images"]["user_agent"], "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"}
    env_name = CFG["images"].get("auth_header_env")
    if env_name and os.environ.get(env_name): headers["Authorization"] = os.environ[env_name]
    attempts = CFG["images"]["retries"] + 1
    error = ""
    for attempt in range(attempts):
        try:
            r = requests.get(url2, headers=headers, timeout=(CFG["images"]["timeout_connect"], CFG["images"]["timeout_read"]),
                             stream=True, allow_redirects=True)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").lower()
            if ctype and not ctype.startswith("image/"):
                r.close(); raise ValueError(f"non-image Content-Type: {ctype}")
            chunks, n = [], 0
            for chunk in r.iter_content(64 * 1024):
                if not chunk: continue
                n += len(chunk)
                if n > CFG["images"]["max_bytes"]: raise ValueError("response exceeded max_bytes")
                chunks.append(chunk)
            r.close()
            ok, msg = _validate_and_save_image(b"".join(chunks), raw_path)
            if not ok: raise ValueError(msg)
            im = Image.open(raw_path); w, h = im.size; im.close()
            return {"product_id": pid, "url": url, "url_hash": uh, "rank": rank, "order": order, "status": "downloaded",
                    "cache_path": str(raw_path), "width": w, "height": h, "lifestyle": lifestyle, "error": ""}
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            if attempt + 1 < attempts: time.sleep(CFG["images"]["backoff"] * (2 ** attempt))
    return {"product_id": pid, "url": url, "url_hash": uh, "rank": rank, "order": order, "status": "failed",
            "cache_path": "", "width": None, "height": None, "lifestyle": lifestyle, "error": error[:500]}

# Use at most N image links per product; set images.max_products for a quick pilot.
DOWNLOAD_LINKS = IMAGE_LINKS.copy()
if CFG["images"]["max_products"] is not None:
    allowed_pids = set(SELECTED["product_id"].head(int(CFG["images"]["max_products"])))
    DOWNLOAD_LINKS = DOWNLOAD_LINKS[DOWNLOAD_LINKS["product_id"].isin(allowed_pids)]
DOWNLOAD_LINKS["_priority"] = DOWNLOAD_LINKS.apply(_image_rank, axis=1)
DOWNLOAD_LINKS = DOWNLOAD_LINKS.sort_values(["product_id", "_priority"], kind="stable")
DOWNLOAD_TASKS = []
for pid, g in DOWNLOAD_LINKS.groupby("product_id", sort=False):
    for rank, (_, row) in enumerate(g.iterrows(), start=1):
        DOWNLOAD_TASKS.append((pid, row["url"], rank, row["order"]))

_image_rows = []
if CFG["images"]["download"] and DOWNLOAD_TASKS and REQUESTS_AVAILABLE:
    with ThreadPoolExecutor(max_workers=CFG["images"]["max_workers"]) as ex:
        futures = [ex.submit(_download_one, t) for t in DOWNLOAD_TASKS]
        for fut in as_completed(futures):
            try: _image_rows.append(fut.result())
            except Exception as e: log(f"unexpected image worker error: {e}", "WARN")
else:
    _image_rows = [_download_one(t) for t in DOWNLOAD_TASKS]
IMAGE_FILES = pd.DataFrame(_image_rows, columns=["product_id", "url", "url_hash", "rank", "order", "status", "cache_path", "width", "height", "lifestyle", "error"])
if len(IMAGE_FILES):
    IMAGE_FILES = IMAGE_FILES.sort_values(["product_id", "rank"], kind="stable").reset_index(drop=True)
    IMAGE_FILES.to_csv(IMAGE_STATUS_PATH, index=False, encoding="utf-8-sig")

# Lightweight foreground crop fallback. OWLv2 in Section 8 can produce a more product-aware crop.
REMBG_REMOVE = None
REMBG_CHECKED = False

def _try_rembg_crop(image_path, out_path):
    """Optional ONNX/rembg subject mask; returns (success, message)."""
    global REMBG_REMOVE, REMBG_CHECKED
    if not CFG["models"].get("use_rembg", False): return False, "disabled"
    if not REMBG_CHECKED:
        REMBG_CHECKED = True
        try:
            from rembg import remove as _remove
            REMBG_REMOVE = _remove
        except Exception as e:
            REMBG_REMOVE = None
            dq("models", "optional rembg unavailable", 1, "info", f"{type(e).__name__}: {e}; using background-difference crop")
    if REMBG_REMOVE is None: return False, "rembg unavailable"
    try:
        with open(image_path, "rb") as f: raw = f.read()
        cutout = REMBG_REMOVE(raw)
        rgba = Image.open(io.BytesIO(cutout)).convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        ys, xs = np.where(alpha > 16)
        if len(xs) < 100: return False, "rembg mask empty"
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()+1), int(ys.min()), int(ys.max()+1)
        bw, bh = x1-x0, y1-y0
        if bw < .12*rgba.width or bh < .12*rgba.height: return False, "rembg mask too small"
        margin = CFG["images"]["crop"]["margin"]
        dx, dy = int(bw*margin), int(bh*margin)
        rgba = rgba.crop((max(0,x0-dx), max(0,y0-dy), min(rgba.width,x1+dx), min(rgba.height,y1+dy)))
        white = Image.new("RGB", rgba.size, "white"); white.paste(rgba, mask=rgba.getchannel("A"))
        white.thumbnail((CFG["images"]["crop"]["target_px"],)*2, Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True); white.save(out_path, "JPEG", quality=95, optimize=True)
        return True, "rembg foreground mask"
    except Exception as e: return False, f"rembg failed: {type(e).__name__}: {e}"

def _basic_foreground_crop(image_path, out_path):
    if not PIL_AVAILABLE: return False, "Pillow unavailable"
    # Prefer an object mask when the optional model is installed; otherwise use deterministic background differencing.
    ok, note = _try_rembg_crop(image_path, out_path)
    if ok: return True, note
    try:
        im = Image.open(image_path).convert("RGB")
        arr = np.asarray(im)
        # Estimate studio-background colour from four corners, then crop largest region unlike that background.
        corners = np.concatenate([arr[:8, :8].reshape(-1, 3), arr[:8, -8:].reshape(-1, 3),
                                  arr[-8:, :8].reshape(-1, 3), arr[-8:, -8:].reshape(-1, 3)], axis=0)
        bg = np.median(corners, axis=0)
        diff = np.sqrt(((arr.astype(np.float32) - bg) ** 2).sum(axis=2))
        # More tolerant than pure-white trimming (works for light grey studio backdrop too).
        mask = diff > 26
        try:
            import cv2
            k = np.ones((5, 5), np.uint8)
            mask_u8 = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_CLOSE, k)
            nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
            if nlab > 1:
                i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y, w, h, area = stats[i]
                if area < 0.015 * mask.size: return False, "foreground too small / uncertain"
                mask = labels == i
                ys, xs = np.where(mask)
            else: ys, xs = np.where(mask)
        except Exception:
            ys, xs = np.where(mask)
        if not len(xs): return False, "no clear foreground; keep original"
        x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        bw, bh = x1-x0, y1-y0
        if bw < 0.15 * im.width or bh < 0.15 * im.height: return False, "foreground mask too small; keep original"
        margin = CFG["images"]["crop"]["margin"]
        dx, dy = int(bw * margin), int(bh * margin)
        crop = im.crop((max(0,x0-dx), max(0,y0-dy), min(im.width,x1+dx), min(im.height,y1+dy)))
        crop.thumbnail((CFG["images"]["crop"]["target_px"], CFG["images"]["crop"]["target_px"]), Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True); crop.save(out_path, "JPEG", quality=94, optimize=True)
        return True, "background-difference crop"
    except Exception as e: return False, f"{type(e).__name__}: {e}"

CROP_ROWS = []
if len(IMAGE_FILES):
    for _, r in IMAGE_FILES.iterrows():
        if r["status"] not in ("downloaded", "cached") or not r["cache_path"]: continue
        crop_path = DIRS["cache_images_crop"] / f"{r['product_id']}_{r['url_hash']}_basic.jpg"
        if not crop_path.exists():
            ok, msg = _basic_foreground_crop(r["cache_path"], crop_path)
            if not ok: crop_path = Path(r["cache_path"]); msg = "original retained: " + msg
        else: ok, msg = True, "cached basic crop"
        CROP_ROWS.append({"product_id": r["product_id"], "url_hash": r["url_hash"], "rank": r["rank"],
                          "source_path": r["cache_path"], "crop_path": str(crop_path), "lifestyle": bool(r["lifestyle"]), "crop_note": msg})
CROP_FILES = pd.DataFrame(CROP_ROWS, columns=["product_id", "url_hash", "rank", "source_path", "crop_path", "lifestyle", "crop_note"])
if len(CROP_FILES): CROP_FILES.to_csv(RES_DIR / "image_crop_manifest.csv", index=False, encoding="utf-8-sig")

# Report broken links and products without usable downloads.
FAILED_IMAGES = IMAGE_FILES[IMAGE_FILES["status"].isin(["failed", "failed_cached"])] if len(IMAGE_FILES) else IMAGE_FILES
dq("images", "downloaded/cached images", int(IMAGE_FILES["status"].isin(["downloaded", "cached"]).sum()) if len(IMAGE_FILES) else 0, "info")
dq("images", "failed image downloads", len(FAILED_IMAGES), "warning" if len(FAILED_IMAGES) else "info",
   "; ".join(FAILED_IMAGES["error"].dropna().astype(str).head(5).tolist()) if len(FAILED_IMAGES) else "")
usable_pids = set(CROP_FILES["product_id"]) if len(CROP_FILES) else set()
products_without_downloaded_image = SELECTED[~SELECTED["product_id"].isin(usable_pids)][["product_id", "product_name", "subcategory"]]
dq("images", "selected products with no usable downloaded/cached image", len(products_without_downloaded_image),
   "warning" if len(products_without_downloaded_image) else "info")
log(f"image links attempted: {len(IMAGE_FILES):,}; usable images: {len(CROP_FILES):,}; failed: {len(FAILED_IMAGES):,}")

if len(CROP_FILES): display(CROP_FILES.head(8))

# %% [markdown]
# ## 8. Sophisticated visual similarity: product-aware crops + SigLIP/DINOv2 embeddings + PDF few-shot examples
#
# **Primary visual model:** SigLIP (a joint image/text embedding model) compares actual product photos with attribute descriptions and
# the PDF example products. **Second visual representation:** DINOv2 captures shape/construction for nearest-example similarity.
# **Optional localization:** OWLv2 (open-vocabulary object detection) crops the target product from lifestyle photos. `rembg`
# is an optional mask fallback. These model packages/checkpoints are not present in every Python environment: this section tries them,
# records failure and continues with local, classical colour/shape features. It never silently claims fallback features are a pretrained model.
#
# First run with pretrained models requires internet or pre-cached model checkpoints. Set `CFG['models']['local_files_only']=True`
# for offline use. `auto` needs `torch` + `transformers`; `open_clip` is an optional alternative.
#
# Embedding steps: (1) choose product image(s) (2) detector crop (when available) (3) encode product + PDF examples (4) cache by
# image-content SHA + model ID (5) match with text descriptions and nearest labelled PDF products.

# %%
# ---------- optional deep-learning models: attempt only as configured; never install implicitly ----------------
TORCH_OK = TRANSFORMERS_OK = False
TORCH = HF = None
PRIMARY_MODEL = PRIMARY_PROCESSOR = None
SECONDARY_MODEL = SECONDARY_PROCESSOR = None
DETECTOR_MODEL = DETECTOR_PROCESSOR = None
PRIMARY_NAME = "none"
MODEL_BACKEND = "classic"
DEVICE = "cpu"

if CFG["models"]["backend"] != "classic":
    try:
        import torch as TORCH
        from transformers import AutoProcessor, AutoModel
        TORCH_OK = TRANSFORMERS_OK = True
        DEVICE = "cuda" if CFG["models"]["device"] == "auto" and TORCH.cuda.is_available() else CFG["models"]["device"]
        if DEVICE == "auto": DEVICE = "cpu"
    except Exception as e:
        log(f"torch/transformers unavailable -> classic visual features ({type(e).__name__}: {e})", "WARN")

if TORCH_OK and CFG["models"]["backend"] in ("auto", "transformers"):
    try:
        _local = CFG["models"]["local_files_only"]
        PRIMARY_PROCESSOR = AutoProcessor.from_pretrained(CFG["models"]["primary_hf_id"], local_files_only=_local)
        PRIMARY_MODEL = AutoModel.from_pretrained(CFG["models"]["primary_hf_id"], local_files_only=_local).to(DEVICE).eval()
        PRIMARY_NAME = CFG["models"]["primary_hf_id"]
        MODEL_BACKEND = "transformers"
        log(f"primary image-text model loaded: {PRIMARY_NAME} on {DEVICE}")
    except Exception as e:
        PRIMARY_MODEL = PRIMARY_PROCESSOR = None
        log(f"primary SigLIP load failed; will attempt fallback ({type(e).__name__}: {e})", "WARN")
        dq("models", "primary SigLIP unavailable", 1, "warning", f"{type(e).__name__}: {e}")

if TORCH_OK and CFG["models"]["secondary"]["enabled"]:
    try:
        _sid = CFG["models"]["secondary"]["hf_id"]
        SECONDARY_PROCESSOR = AutoProcessor.from_pretrained(_sid, local_files_only=CFG["models"]["local_files_only"])
        SECONDARY_MODEL = AutoModel.from_pretrained(_sid, local_files_only=CFG["models"]["local_files_only"]).to(DEVICE).eval()
        log(f"secondary shape model loaded: {_sid}")
    except Exception as e:
        SECONDARY_MODEL = SECONDARY_PROCESSOR = None
        log(f"DINOv2 unavailable (optional); continuing without second embedding ({type(e).__name__}: {e})", "WARN")
        dq("models", "DINOv2 unavailable", 1, "warning", f"{type(e).__name__}: {e}")

if PRIMARY_MODEL is not None:
    MODEL_BACKEND = "transformers"
else:
    # open_clip is another optional SigLIP backend (requires torch + open_clip_torch and cached/downloadable weights)
    if CFG["models"]["backend"] in ("auto", "open_clip"):
        try:
            import open_clip
            if TORCH is None: import torch as TORCH
            _m, _, _pre = open_clip.create_model_and_transforms(CFG["models"]["open_clip_name"],
                  pretrained=CFG["models"]["open_clip_pretrained"])
            PRIMARY_MODEL, PRIMARY_PROCESSOR = _m.to(DEVICE).eval(), _pre
            PRIMARY_NAME = CFG["models"]["open_clip_name"] + "/" + CFG["models"]["open_clip_pretrained"]
            MODEL_BACKEND = "open_clip"
            log(f"open_clip image-text model loaded: {PRIMARY_NAME}")
        except Exception as e:
            log(f"open_clip not available; classic image descriptors will be used ({type(e).__name__}: {e})", "WARN")

if PRIMARY_MODEL is None and MODEL_BACKEND != "open_clip":
    MODEL_BACKEND = "dinov2" if SECONDARY_MODEL is not None else "classic"
    if SECONDARY_MODEL is not None: PRIMARY_NAME = CFG["models"]["secondary"]["hf_id"]

# ---------- optional product object detector (OWL-ViT v2, target-aware crop) --------------------
if TORCH_OK and CFG["models"]["detector"]["enabled"] and MODEL_BACKEND == "transformers":
    try:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        _did = CFG["models"]["detector"]["hf_id"]
        DETECTOR_PROCESSOR = Owlv2Processor.from_pretrained(_did, local_files_only=CFG["models"]["local_files_only"])
        DETECTOR_MODEL = Owlv2ForObjectDetection.from_pretrained(_did, local_files_only=CFG["models"]["local_files_only"]).to(DEVICE).eval()
        log(f"open-vocabulary product detector loaded: {_did}")
    except Exception as e:
        DETECTOR_MODEL = DETECTOR_PROCESSOR = None
        log(f"OWLv2 unavailable; use primary product shots / background crops ({type(e).__name__}: {e})", "WARN")
        dq("models", "OWLv2 unavailable", 1, "warning", f"{type(e).__name__}: {e}")

# ---------- image file index, optional OWLv2 target crops --------------------------------------
IMAGE_RECORDS = []
if len(CROP_FILES):
    for _, r in CROP_FILES.iterrows():
        row = r.to_dict()
        row["detector_score"] = np.nan
        row["detector_label"] = ""
        row["final_path"] = row["crop_path"]
        IMAGE_RECORDS.append(row)


def _detect_product_crop(record):
    """Return target-aware crop if OWLv2 finds the requested product; otherwise None."""
    if DETECTOR_MODEL is None or not PIL_AVAILABLE: return None
    pid, rule_key = record["product_id"], SELECTED.loc[SELECTED["product_id"] == record["product_id"], "subcategory"].iloc[0]
    rule_key = CFG["targets"][rule_key]["rule_key"]
    query = DETECTOR_QUERIES.get(rule_key, [NOUNS.get(rule_key, "product")])
    try:
        im = Image.open(record["source_path"]).convert("RGB")
        inputs = DETECTOR_PROCESSOR(text=[query], images=im, return_tensors="pt").to(DEVICE)
        with TORCH.no_grad(): outputs = DETECTOR_MODEL(**inputs)
        sizes = TORCH.tensor([[im.height, im.width]], device=DEVICE)
        results = DETECTOR_PROCESSOR.post_process_object_detection(outputs, threshold=CFG["models"]["detector"]["min_score"], target_sizes=sizes)[0]
        if len(results["scores"]) == 0: return None
        i = int(results["scores"].argmax().item())
        score = float(results["scores"][i].item())
        box = results["boxes"][i].detach().cpu().numpy().astype(int).tolist()
        x0, y0, x1, y1 = box
        if x1 - x0 < 0.10*im.width or y1 - y0 < 0.10*im.height: return None
        margin = CFG["images"]["crop"]["margin"]
        dx, dy = int((x1-x0)*margin), int((y1-y0)*margin)
        crop = im.crop((max(0,x0-dx), max(0,y0-dy), min(im.width,x1+dx), min(im.height,y1+dy)))
        crop.thumbnail((CFG["images"]["crop"]["target_px"],)*2, Image.Resampling.LANCZOS)
        p = DIRS["cache_images_crop"] / f"{pid}_{record['url_hash']}_owlv2.jpg"
        crop.save(p, "JPEG", quality=95, optimize=True)
        return {"final_path": str(p), "detector_score": score, "detector_label": ",".join(query)}
    except Exception as e:
        dq("images", "OWLv2 crop failed", 1, "info", f"product={pid}; {type(e).__name__}: {e}")
        return None

# Try the detector for the top ranked product image per product; use alternate source images only when primary is lifestyle.
if DETECTOR_MODEL is not None and IMAGE_RECORDS:
    _det_count = 0
    for rec in IMAGE_RECORDS:
        if rec["rank"] > 1 and not rec["lifestyle"]: continue
        result = _detect_product_crop(rec)
        if result:
            rec.update(result); _det_count += 1
    log(f"OWLv2 produced target-aware crops for {_det_count:,} image(s); other images use basic/original crop")

# Re-sort images: clean shot first; high detector score next; then rank. Use up to 3 views/product.
if IMAGE_RECORDS:
    IMAGE_RECORDS.sort(key=lambda r: (r["product_id"], bool(r["lifestyle"]), -(r["detector_score"] if pd.notna(r["detector_score"]) else 0), r["rank"]))
IMAGE_INDEX = defaultdict(list)
for r in IMAGE_RECORDS:
    if r["final_path"] and Path(r["final_path"]).exists(): IMAGE_INDEX[r["product_id"]].append(r)
for pid in list(IMAGE_INDEX): IMAGE_INDEX[pid] = IMAGE_INDEX[pid][:CFG["images"]["max_images_per_product"]]
PRODUCT_IMAGE_PATHS = {pid: [r["final_path"] for r in rows] for pid, rows in IMAGE_INDEX.items()}
IMAGE_RECORDS_DF = pd.DataFrame(IMAGE_RECORDS)
if len(IMAGE_RECORDS_DF): IMAGE_RECORDS_DF.to_csv(RES_DIR / "product_image_crop_manifest.csv", index=False, encoding="utf-8-sig")

# ---------- image/text embedding backend -------------------------------------------------------
def _l2norm(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)

def _to_numpy(x):
    if hasattr(x, "detach"): return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def _classic_descriptor(image_path):
    """Fallback only: HOG-like shape + HSV histogram. It is not a pretrained semantic embedding."""
    from PIL import Image
    im = Image.open(image_path).convert("RGB").resize((96, 96))
    arr = np.asarray(im)
    try:
        import cv2
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hog = cv2.HOGDescriptor((96,96), (24,24), (12,12), (12,12), 9)
        shape = hog.compute(gray).reshape(-1).astype(np.float32)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [18,4], [0,180,0,256]).flatten().astype(np.float32)
    except Exception:
        from skimage.feature import hog as sk_hog
        gray = np.asarray(im.convert("L"))
        shape = sk_hog(gray, orientations=9, pixels_per_cell=(12,12), cells_per_block=(2,2), feature_vector=True).astype(np.float32)
        hist = np.histogramdd(arr.reshape(-1,3), bins=(6,6,6), range=((0,256),(0,256),(0,256)))[0].flatten().astype(np.float32)
    return _l2norm(np.concatenate([_l2norm(shape), _l2norm(hist)]))

class VisualEncoder:
    def __init__(self):
        self.name = PRIMARY_NAME if PRIMARY_MODEL is not None else "classic-hog-hsv"
        self.has_text = PRIMARY_MODEL is not None
    def encode_images(self, paths, include_secondary=True):
        paths = list(paths)
        if not paths: return np.empty((0,0), dtype=np.float32)
        if PRIMARY_MODEL is None and SECONDARY_MODEL is None:
            return np.vstack([_classic_descriptor(p) for p in paths])
        primary, secondary = [], []
        bs = int(CFG["models"]["batch_size"])
        for start in range(0, len(paths), bs):
            batch_paths = paths[start:start+bs]
            ims = [Image.open(p).convert("RGB") for p in batch_paths]
            if PRIMARY_MODEL is not None:
                with TORCH.no_grad():
                    if MODEL_BACKEND == "transformers":
                        inp = PRIMARY_PROCESSOR(images=ims, return_tensors="pt").to(DEVICE)
                        out = PRIMARY_MODEL.get_image_features(**inp)
                        if hasattr(out, "pooler_output"): out = out.pooler_output
                        primary.append(_l2norm(_to_numpy(out)))
                    else:
                        batch = TORCH.stack([PRIMARY_PROCESSOR(im) for im in ims]).to(DEVICE)
                        primary.append(_l2norm(_to_numpy(PRIMARY_MODEL.encode_image(batch))))
            if include_secondary and SECONDARY_MODEL is not None:
                with TORCH.no_grad():
                    inp2 = SECONDARY_PROCESSOR(images=ims, return_tensors="pt").to(DEVICE)
                    out2 = SECONDARY_MODEL(**inp2)
                    vec = out2.last_hidden_state[:, 0, :] if hasattr(out2, "last_hidden_state") else out2[0][:, 0, :]
                    secondary.append(_l2norm(_to_numpy(vec)))
            for im in ims: im.close()
        if primary and secondary:
            w = float(CFG["models"]["secondary"]["weight"])
            return _l2norm(np.concatenate([np.vstack(primary)*math.sqrt(1-w), np.vstack(secondary)*math.sqrt(w)], axis=1))
        if primary: return _l2norm(np.vstack(primary))
        if secondary: return _l2norm(np.vstack(secondary))
        return np.vstack([_classic_descriptor(p) for p in paths])
    def encode_texts(self, texts):
        if PRIMARY_MODEL is None: return None
        if MODEL_BACKEND == "transformers":
            with TORCH.no_grad():
                inputs = PRIMARY_PROCESSOR(text=list(texts), padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                out = PRIMARY_MODEL.get_text_features(**inputs)
                if hasattr(out, "pooler_output"): out = out.pooler_output
                return _l2norm(_to_numpy(out))
        try:
            import open_clip
            toks = open_clip.get_tokenizer(CFG["models"]["open_clip_name"])(list(texts)).to(DEVICE)
            with TORCH.no_grad(): return _l2norm(_to_numpy(PRIMARY_MODEL.encode_text(toks)))
        except Exception as e:
            log(f"text encoder unavailable: {e}", "WARN"); return None

ENCODER = VisualEncoder()
log(f"visual backend = {MODEL_BACKEND}; encoder = {ENCODER.name}; text embedding = {ENCODER.has_text}; detector = {DETECTOR_MODEL is not None}")

# ---------- cache deep/classical embeddings by image bytes + encoder ID -------------------------
EMBED_CACHE_FILE = DIRS["cache_embeddings"] / ("embeddings_" + hashlib.md5((ENCODER.name + str(CFG["models"]["secondary"])).encode()).hexdigest()[:10] + ".npz")
_EMBED_CACHE = {}
if EMBED_CACHE_FILE.exists():
    try:
        _z = np.load(EMBED_CACHE_FILE, allow_pickle=False)
        _EMBED_CACHE = {k: _z[k] for k in _z.files}
        log(f"loaded {len(_EMBED_CACHE):,} cached image embeddings")
    except Exception as e: log(f"embedding cache not reusable ({e}); rebuild", "WARN")

def _file_content_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()[:32]

def cached_encode(paths, include_secondary=True):
    paths = list(paths)
    if not paths: return np.empty((0,0), dtype=np.float32)
    keys = [_file_content_hash(p) for p in paths]
    missing_pos = [i for i,k in enumerate(keys) if k not in _EMBED_CACHE]
    if missing_pos:
        vecs = ENCODER.encode_images([paths[i] for i in missing_pos], include_secondary=include_secondary)
        for i, v in zip(missing_pos, vecs): _EMBED_CACHE[keys[i]] = np.asarray(v, dtype=np.float32)
        # Atomic update avoids half-written cache after interruption.
        tmp = EMBED_CACHE_FILE.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **_EMBED_CACHE)
        tmp.replace(EMBED_CACHE_FILE)
    return np.vstack([_EMBED_CACHE[k] for k in keys])

# Encode each downloaded product view; PDF example vectors share the same cache.
PRODUCT_IMAGE_EMBEDDINGS = {}
_product_paths_flat, _product_key_map = [], []
for pid, paths in PRODUCT_IMAGE_PATHS.items():
    for p in paths:
        _product_paths_flat.append(p); _product_key_map.append((pid, p))
if _product_paths_flat:
    _pv = cached_encode(_product_paths_flat)
    for key, vec in zip(_product_key_map, _pv): PRODUCT_IMAGE_EMBEDDINGS[key] = vec

PDF_IMAGE_EMBEDDINGS = {}
if len(PDF_EXAMPLES):
    _pdf_paths = [p for p in PDF_EXAMPLES["image_path"] if Path(p).exists()]
    if _pdf_paths:
        _fv = cached_encode(_pdf_paths)
        for p, vec in zip(_pdf_paths, _fv): PDF_IMAGE_EMBEDDINGS[p] = vec

def product_visual_vector(product_id):
    rows = IMAGE_INDEX.get(product_id, [])
    vs = [PRODUCT_IMAGE_EMBEDDINGS[(product_id, r["final_path"])] for r in rows if (product_id, r["final_path"]) in PRODUCT_IMAGE_EMBEDDINGS]
    if not vs: return None
    # Average the top two views so secondary angles do not outweigh a clean primary packshot.
    return _l2norm(np.mean(np.vstack(vs[:2]), axis=0))

PRODUCT_VISUAL = {p: product_visual_vector(p) for p in SELECTED["product_id"]}
PRODUCT_VISUAL = {p: v for p,v in PRODUCT_VISUAL.items() if v is not None}

# ---------- text prompt vectors for attributes + design-style labels ---------------------------
PROMPT_TEXTS, PROMPT_KEYS = [], []
for rule_key, attr_map in IMAGE_PROMPTS.items():
    for attr, spec in attr_map.items():
        for value, desc in spec.get("values", {}).items():
            if spec.get("kind") == "binary" and isinstance(desc, (list, tuple)):
                for polarity, d in zip(("present", "absent"), desc):
                    PROMPT_KEYS.append((rule_key, attr, value + "::" + polarity, d))
                    PROMPT_TEXTS.append(PROMPT_TEMPLATE.format(noun=NOUNS.get(rule_key, "product"), desc=d))
            else:
                descs = desc if isinstance(desc, (list,tuple)) else [desc]
                for d in descs:
                    PROMPT_KEYS.append((rule_key, attr, value, d))
                    PROMPT_TEXTS.append(PROMPT_TEMPLATE.format(noun=NOUNS.get(rule_key, "product"), desc=d))
for rule_key, value_map in STYLE_PROMPTS.items():
    for value, descs in value_map.items():
        for d in descs:
            PROMPT_KEYS.append((rule_key, "Design style", value, d))
            PROMPT_TEXTS.append(PROMPT_TEMPLATE.format(noun=NOUNS.get(rule_key, "product"), desc=d))
TEXT_EMBEDDINGS = ENCODER.encode_texts(PROMPT_TEXTS) if PROMPT_TEXTS and ENCODER.has_text else None
PROMPT_VECTOR_MAP = defaultdict(list)
if TEXT_EMBEDDINGS is not None:
    for k, v in zip(PROMPT_KEYS, TEXT_EMBEDDINGS): PROMPT_VECTOR_MAP[k[:3]].append(v)

def _image_prompt_scores(product_id, rule_key, attribute, allowed_values, prompt_specs=None):
    """Average prompt similarities per allowed value; return normalized value probabilities + similarity margin."""
    if TEXT_EMBEDDINGS is None: return {}, {"model": "no text encoder"}
    vec = PRODUCT_VISUAL.get(product_id)
    if vec is None: return {}, {"model": "no product image embedding"}
    spec = prompt_specs or IMAGE_PROMPTS.get(rule_key, {}).get(attribute, {})
    if attribute == "Design style" and not spec:
        spec = {"values": STYLE_PROMPTS.get(rule_key, {})}
    vals = list(spec.get("values", {}).keys())
    if not vals: return {}, {}
    if spec.get("kind") == "binary":
        out = {}
        t = float(CFG["models"]["zero_shot_temperature"])
        raw = []
        for val in vals:
            pos = PROMPT_VECTOR_MAP.get((rule_key, attribute, val + "::present"), [])
            neg = PROMPT_VECTOR_MAP.get((rule_key, attribute, val + "::absent"), [])
            if pos and neg:
                delta = float(np.mean(np.vstack(pos) @ vec) - np.mean(np.vstack(neg) @ vec))
                prob = 1.0 / (1.0 + math.exp(float(np.clip(-delta * t, -50, 50))))
                out[val] = prob; raw.append(delta)
        return out, {"top_cosine": max(raw) if raw else 0.0, "margin": max(raw) if raw else 0.0, "prompt_count": len(raw), "model": ENCODER.name}
    sims = []
    for val in vals:
        pvs = PROMPT_VECTOR_MAP.get((rule_key, attribute, val), [])
        if pvs: sims.append((val, float(np.mean(np.vstack(pvs) @ vec))))
    if not sims: return {}, {}
    raw = np.array([x[1] for x in sims], dtype=float)
    t = float(CFG["models"]["zero_shot_temperature"])
    z = (raw - raw.max()) * t
    ex = np.exp(np.clip(z, -50, 0)); probs = ex / max(ex.sum(), 1e-12)
    return dict(zip([x[0] for x in sims], probs)), {"top_cosine": float(raw.max()), "margin": float(np.sort(raw)[-1] - np.sort(raw)[-2]) if len(raw)>1 else 0.0,
                                                     "prompt_count": len(sims), "model": ENCODER.name}

# ---------- PDF few-shot style model: nearest neighbours + out-of-fold LR ----------------------
STYLE_MODELS = {}
STYLE_LOGREG = {}

def _rule_allowed_styles(rule_key):
    target = next((t for t,s in CFG["targets"].items() if s["rule_key"] == rule_key), None)
    if target is None: return []
    return ALLOWED[MATRIX_CATEGORY[target]].get("Design style", [])

def _style_features(rule_key, allow_transfer=True):
    if not len(PDF_EXAMPLES) or not PDF_IMAGE_EMBEDDINGS: return None
    own = PDF_EXAMPLES[PDF_EXAMPLES["rule_key"] == rule_key]
    if len(own): source = own
    elif allow_transfer and CFG["pdf"]["transfer_to_categories_without_examples"]:
        source = PDF_EXAMPLES[PDF_EXAMPLES["style_value"].isin(_rule_allowed_styles(rule_key))]
    else: return None
    rows, labels, cats, pages = [], [], [], []
    for _, r in source.iterrows():
        p = r["image_path"]
        if p not in PDF_IMAGE_EMBEDDINGS or not r["style_value"]: continue
        rows.append(PDF_IMAGE_EMBEDDINGS[p]); labels.append(r["style_value"]); cats.append(r["rule_key"]); pages.append(int(r["pdf_page"]))
    if not rows: return None
    return np.vstack(rows), np.array(labels), np.array(cats), bool(len(own)), np.array(pages)

def _style_probabilities(rule_key, product_vec):
    if product_vec is None: return {}, {"status": "no product image"}
    data = _style_features(rule_key)
    if data is None: return {}, {"status": "no PDF examples"}
    X, y, cats, has_own, pages = data
    target = next(t for t,s in CFG["targets"].items() if s["rule_key"] == rule_key)
    valid = set(ALLOWED[MATRIX_CATEGORY[target]].get("Design style", []))
    mask = np.array([lab in valid for lab in y])
    X, y, cats, pages = X[mask], y[mask], cats[mask], pages[mask]
    if not len(y): return {}, {"status": "no allowed PDF labels"}
    # Cosine nearest neighbours: image-to-image similarity to actual specialist-labelled products.
    sims = _l2norm(X) @ _l2norm(product_vec.reshape(1,-1)).ravel()
    order = np.argsort(-sims)
    k = min(int(CFG["pdf"]["knn_k"]), len(order))
    counts = defaultdict(float)
    for j in order[:k]: counts[y[j]] += max(0.01, float(sims[j]))
    knn = {v: counts[v] / max(sum(counts.values()),1e-12) for v in valid}
    max_sim = float(sims[order[0]])
    nearest_label = str(y[order[0]])
    # Optional logistic regression probabilities; fit only when at least two labels are represented.
    lr = {}
    if len(set(y)) > 1:
        try:
            from sklearn.linear_model import LogisticRegression
            if rule_key not in STYLE_LOGREG:
                clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=17)
                clf.fit(_l2norm(X), y)
                STYLE_LOGREG[rule_key] = clf
            clf = STYLE_LOGREG[rule_key]
            pr = clf.predict_proba(_l2norm(product_vec.reshape(1,-1)))[0]
            lr = dict(zip(clf.classes_, pr))
        except Exception: lr = {}
    model = {v: (CFG["pdf"]["fewshot_mix_lr"] * lr.get(v, 0) + (1-CFG["pdf"]["fewshot_mix_lr"]) * knn.get(v,0))
             if lr else knn.get(v,0) for v in valid}
    # Cross-source examples are useful weak similarity evidence, not a product-category-specific supervised model.
    if not has_own:
        factor = CFG["pdf"]["transfer_reliability_factor"]
        model = {v: p for v,p in model.items()}
    return model, {"status": "few-shot" if has_own else "transferred similarity", "has_own_examples": has_own,
                   "n_examples": len(y), "nearest_similarity": max_sim, "nearest_style": nearest_label,
                   "nearest_pdf_page": int(pages[order[0]]) if len(pages) else None,
                   "source_categories": sorted(set(cats.tolist()))}

# Internal held-out reliability estimate: leave-one-out nearest-neighbour accuracy for each source group.
STYLE_RELIABILITY = {}
if len(PDF_EXAMPLES) and PDF_IMAGE_EMBEDDINGS:
    for rk in CFG["pdf"]["page_ranges"]:
        data = _style_features(rk)
        if data is None: continue
        X, y, cats, has_own, pages = data
        if not has_own: continue
        X = _l2norm(X)
        correct, n = 0, 0
        for i in range(len(y)):
            sims = X @ X[i]
            sims[i] = -1
            j = int(np.argmax(sims))
            correct += int(y[j] == y[i]); n += 1
        acc = correct / max(n,1)
        STYLE_RELIABILITY[rk] = float(np.clip(acc, CFG["pdf"]["reliability_floor"], CFG["pdf"]["reliability_cap"]))

STYLE_MODEL_SUMMARY = pd.DataFrame([{"rule_key": rk, "loo_nearest_accuracy": val, "pdf_examples": int((PDF_EXAMPLES.rule_key==rk).sum()) if len(PDF_EXAMPLES) else 0}
                                    for rk,val in STYLE_RELIABILITY.items()])
display(STYLE_MODEL_SUMMARY)

# %% [markdown]
# ## 9. Attribute assignment, confidence and evidence
# This is **repeatable rule-based evidence aggregation**, not hand-assigned product results. The allowed-value table defines every
# output value. For each selected product, every allowed category attribute gets exactly one row (a multi-value attribute joins
# only allowed options with the configured separator). If evidence is not sufficient, the explicit `Not determinable` sentinel
# is written rather than an empty cell or an invented value.
#
# **Confidence formula:** take the strongest support per evidence family for each candidate value (so duplicate text fields do not
# inflate a score), combine independent evidence families with noisy-OR, subtract `conflict_penalty × strongest competing value`,
# then apply the style-specific ceiling. High ≥ 0.75; Medium ≥ 0.50; Low ≥ 0.30; lower → Not determinable. A direct structured
# value is normally High. Text/visual similarity has lower reliability. Style is capped at Medium unless near-identical to a
# specialist-labelled PDF example; a category with no own examples (Vase) is capped at Low.

# %%
# ---------- context and value helpers ---------------------------------------------------------
NOT_DETERMINABLE = CFG["assignment"]["sentinel_not_determinable"]
NONE_DETECTED = CFG["assignment"]["sentinel_none_detected"]

# Direct existing-attribute names; values must still equal a valid allowed value to be accepted as direct evidence.
DIRECT_ATTRIBUTE_ALIASES = {
  "Design style": ["Design style", "Shopify Design Style", "Style"],
  "Material": ["Material", "Shopify Material"],
  "Material & look": ["Material & look", "Shopify Material"],
  "Material & look (table top)": ["Material & look (table top)", "Shopify Material", "Material"],
  "Seat Material & look": ["Seat Material & look", "Shopify Sitzbezug", "Shopify Sitz", "Shopify Material"],
  "Frame Material & look": ["Frame Material & look", "Shopify Gestell"],
  "Colour & finish": ["Colour & finish", "Shopify Farbe", "Farbe", "ColorName"],
  "Colour Group": ["Colour Group", "Shopify Farbe", "Farbe", "ColorName"],
  "Size (volume)": ["Size (volume)", "Fassungsvermögen"],
  "Size (seat width)": ["Size (seat width)", "Sitzbreite"],
}

def _category_rule_key(target): return CFG["targets"][target]["rule_key"]

def _product_row(pid):
    # product_id is unique by validation; O(1) lookup via a precomputed dict below.
    return PRODUCT_BY_ID.get(pid, {})

PRODUCT_BY_ID = {r["product_id"]: r for r in SELECTED.to_dict("records")}
ATTRS_BY_ID = defaultdict(list)
for _, _ar in SELECTED_ATTR.iterrows():
    ATTRS_BY_ID[_ar["product_id"]].append((_ar["attr_name"], _ar["attr_value"], _ar.get("attr_set", "")))

def _context_entries(pid):
    p = _product_row(pid)
    out = []
    def add(field, value, source, label=None):
        if value is None or (isinstance(value, float) and np.isnan(value)): return
        v = clean_text(value)
        if not isinstance(v, str): return
        out.append({"field": str(field), "value": v, "source_type": source, "source_label": label or str(field)})
    add("name", p.get("product_name"), "product_text", "product name")
    add("variant", p.get("variant"), "product_text", "product variant")
    add("color_name", p.get("color_name"), "product_attribute", "product colour field")
    add("material", p.get("material"), "product_attribute", "product material field")
    for f in ["top_category", "main_category", "sub_category"]:
        add(f, p.get(f), "product_attribute", f)
    for f, value in p.items():
        if f not in {"product_id", "product_name", "variant", "color_name", "material", "top_category", "main_category", "sub_category",
                     "subcategory", "matrix_category", "sub_category_key", "image_urls", "existing_attribute_count", "image_link_count",
                     "length_cm", "depth_cm", "height_cm", "weight_g"} and not str(f).startswith("_"):
            add(str(f), value, "product_attribute", str(f))
    for name, value, aset in ATTRS_BY_ID.get(pid, []):
        add(name, value, "product_attribute", f"ItemAttribute: {name}" + (f" ({aset})" if isinstance(aset, str) and aset else ""))
    # Name-tail helper: after the last comma is often the specific finish/colour, and is useful when Shopify colour is blank.
    name = p.get("product_name") or ""
    if "," in name: add("name_tail", name.rsplit(",", 1)[-1], "product_text", "product name (last phrase)")
    return out

CONTEXT_CACHE = {pid: _context_entries(pid) for pid in SELECTED["product_id"]}

def _folded_context(pid):
    return [(e, fold_text(e["value"])) for e in CONTEXT_CACHE.get(pid, [])]

def _allowed_for(target, attr):
    return ALLOWED[MATRIX_CATEGORY[target]].get(attr, [])

def _exact_allowed(raw_value, allowed_values):
    if raw_value is None: return None
    rv = norm_key(raw_value)
    for av in allowed_values:
        if norm_key(av) == rv: return av
    return None

def _value_is_allowed(value, allowed_values): return value in allowed_values

def _add_candidate(bucket, value, reliability, source_type, source_label, evidence, allowed_values, meta=None):
    if value not in allowed_values:
        dq("assignment", "candidate value rejected because not in allowed values", 1, "warning", f"value={value!r}; allowed={allowed_values}")
        return
    bucket[value].append({"reliability": float(np.clip(reliability, 0.0, 0.99)), "source_type": source_type,
                          "source_label": source_label, "evidence": str(evidence)[:300], "meta": meta or {}})

def _candidate_reliability(source_type, field=None, multiplier=1.0):
    key = {"product_attribute": "product_attribute", "product_text": "product_text", "dimensions": "dimensions",
           "image": "image_zeroshot", "image_colour": "image_colour", "pdf_example": "style_zeroshot",
           "style_text_hint": "style_text_hint", "column_map": "column_map", "generic_text": "generic_text",
           "estimate": "estimate"}.get(source_type, "generic_text")
    return float(CFG["assignment"]["reliability"].get(key, 0.50)) * multiplier

# ---------- configurable new-column mappings (price, tier, ...) --------------------------------
def _column_mapping_candidates(target, attribute, pid, allowed_values, bucket):
    p = _product_row(pid)
    for rule in ATTRIBUTE_FROM_COLUMN:
        cats = rule.get("category", "*")
        if isinstance(cats, str): cats = [cats]
        if "*" not in cats and target not in cats and MATRIX_CATEGORY[target] not in cats: continue
        if norm_key(rule.get("attribute", "")) != norm_key(attribute): continue
        col = rule.get("column")
        if not col or col not in p: continue
        raw = p.get(col)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)): continue
        mode = rule.get("mode", "exact")
        outval = None
        if mode == "exact": outval = _exact_allowed(raw, allowed_values)
        elif mode == "map":
            mp = rule.get("map", {})
            outval = next((v for k,v in mp.items() if norm_key(k) == norm_key(raw)), None)
            outval = _exact_allowed(outval, allowed_values) if outval is not None else None
        elif mode == "bins":
            x = parse_number(raw); edges = rule.get("edges", []); labels = rule.get("labels", [])
            if pd.notna(x) and len(labels) == len(edges)+1:
                idx = next((i for i,e in enumerate(edges) if x <= e), len(edges))
                outval = _exact_allowed(labels[idx], allowed_values)
        if outval:
            _add_candidate(bucket, outval, _candidate_reliability("column_map"), "product_attribute", f"product column: {col}",
                           f"Configured {mode} mapping from {col}={raw!r} to allowed value {outval!r}.", allowed_values)

# ---------- direct structured attributes (only exact, relevant name + allowed value) ------------
def _direct_attribute_candidates(target, attribute, pid, allowed_values, bucket):
    names = DIRECT_ATTRIBUTE_ALIASES.get(attribute, [attribute])
    allowed_names = {norm_key(n) for n in names}
    for name, value, aset in ATTRS_BY_ID.get(pid, []):
        if norm_key(name) not in allowed_names: continue
        # Values with several distinct values can only be kept as a multi assignment if each is explicitly allowed.
        parts = re.split(r"\s*[|;]\s*", str(value))
        for part in parts:
            v = _exact_allowed(part, allowed_values)
            if v:
                _add_candidate(bucket, v, _candidate_reliability("product_attribute"), "product_attribute",
                               f"ItemAttribute: {name}", f"Existing structured attribute {name}={part!r} exactly matches an allowed value.", allowed_values)
    # Direct exact DWH field values only when the input field name is meaningfully the same as the output attribute.
    p = _product_row(pid)
    for field in ["material", "color_name"]:
        if norm_key(field) in allowed_names:
            v = _exact_allowed(p.get(field), allowed_values)
            if v:
                _add_candidate(bucket, v, _candidate_reliability("product_attribute"), "product_attribute", field,
                               f"Product field {field} exactly matches allowed value {v!r}.", allowed_values)

# ---------- keyword matching on configured fields ---------------------------------------------
def _keyword_candidates(target, attribute, pid, allowed_values, bucket):
    rk = _category_rule_key(target)
    rule = KEYWORD_RULES.get(rk, {}).get(attribute)
    if not rule: return
    contexts = _folded_context(pid)
    wanted = {norm_key(f) for f in rule.get("fields", [])}
    available = [(e, text) for e,text in contexts if norm_key(e["field"]) in wanted]
    # If a field is explicitly listed, use it; do not silently search unrelated fields.
    matched = []
    for pattern_row in rule.get("table", []):
        value, pattern = pattern_row[0], pattern_row[1]
        opts = pattern_row[2] if len(pattern_row) > 2 and isinstance(pattern_row[2], dict) else {}
        try: rx = re.compile(pattern, re.I)
        except re.error as e:
            dq("rules", "invalid keyword regex", 1, "warning", f"{rk}/{attribute}: {pattern}: {e}"); continue
        for e, text in available:
            m = rx.search(text)
            if not m: continue
            matched.append({"value": value, "field": e, "match": m.group(0), "opts": opts, "pattern": pattern})
    seen_values = {m["value"] for m in matched if m["value"]}
    # Apply explicit negation / override rules after evaluating the configured evidence fields.
    for m in matched:
        value = m["value"]
        opts = m["opts"]
        if value is None: continue
        if any(v in seen_values for v in opts.get("unless", [])): continue
        rel_factor = float(opts.get("rel", rule.get("reliability_factor", 1.0)))
        e = m["field"]
        src = e["source_type"]
        source_type = "product_attribute" if src == "product_attribute" else "product_text"
        rel = _candidate_reliability(source_type, e["field"], rel_factor)
        # First-match priority (e.g. Leather over a colour word in the same product description).
        if rule.get("first_match_wins"):
            higher = [x for x in matched if x["value"] and x["value"] != value and x["pattern"] != m["pattern"]]
            if higher:
                # Respect the order in the configured table: earlier pattern has precedence.
                all_patterns = [x[1] for x in rule.get("table", []) if x[0] is not None]
                if all_patterns.index(m["pattern"]) > min(all_patterns.index(x["pattern"]) for x in higher): continue
        _add_candidate(bucket, value, rel, source_type, e["source_label"],
                       f"Matched {m['match']!r} in {e['source_label']} using configured {attribute} keyword rule.", allowed_values)

    # Mixed-material signal requires two distinct material families in a designated structured field.
    mixed_value = rule.get("mixed_value")
    if mixed_value and mixed_value in allowed_values:
        fields = {norm_key(f) for f in rule.get("mixed_fields", [])}
        mat_fields = [(e,t) for e,t in contexts if norm_key(e["field"]) in fields]
        material_groups = {
            "wood": r"eich|oak|esche|birke|buche|walnuss|nussbaum|wood|holz|teak|bambus|rattan",
            "metal": r"metall|metal|stahl|steel|aluminium|messing|brass|chrom|kupfer|eisen",
            "glass": r"glas|glass|kristall", "ceramic": r"keramik|ceramic|porzellan|porcelain|steinzeug",
            "stone": r"marmor|marble|travertin|naturstein|granit|beton|concrete",
            "plastic": r"kunststoff|plastic|polypropylen|acryl|silikon|resin",
            "textile": r"stoff|textil|fabric|leinen|linen|baumwolle|cotton",
        }
        detected = set()
        for _, text in mat_fields:
            for group, pattern in material_groups.items():
                if re.search(pattern, text, re.I): detected.add(group)
        if len(detected) >= 2:
            _add_candidate(bucket, mixed_value, _candidate_reliability("product_attribute"), "product_attribute", "material fields",
                           "Two distinct material families are explicitly present: " + ", ".join(sorted(detected)) + ".", allowed_values)

# ---------- colour text and colour image -------------------------------------------------------
def _colour_parse(text):
    text = fold_text(text)
    hits = []
    occupied = []
    for pattern, group, flags in COLOUR_LEXICON:
        try: rx = re.compile(pattern, re.I)
        except re.error: continue
        for m in rx.finditer(text):
            if any(not (m.end() <= a or m.start() >= b) for a,b in occupied): continue
            occupied.append((m.start(), m.end()))
            if group is not None: hits.append((group, m.group(0), flags))
            elif "transparent" in flags: hits.append(("transparent", m.group(0), flags))
            elif "metallic" in flags: hits.append(("metallic", m.group(0), flags))
    if not hits: return []
    counts = Counter()
    terms = defaultdict(list)
    for group, term, flags in hits:
        counts[group] += 1; terms[group].append(term)
    total = sum(counts.values())
    probs = {g: n / total for g,n in counts.items()}
    return [(g, probs[g], ", ".join(terms[g])) for g in counts]

def _map_colour_group(rule_key, attribute, group, allowed_values, group_probs):
    # For Colour Group attributes, mapped labels match the allowed list exactly (excluding the special 'transparent'/'metallic').
    if group in allowed_values: return {group: group_probs}
    mapping = COLOUR_VALUE_MAP.get(rule_key, {}).get(attribute, {}).get(group, {})
    return {v: group_probs * prob for v, prob in mapping.items() if v in allowed_values}

def _colour_text_candidates(target, attribute, pid, allowed_values, bucket):
    rk = _category_rule_key(target)
    fields = [norm_key(f) for f in COLOUR_TEXT_FIELDS]
    contexts = [(e,t) for e,t in _folded_context(pid) if norm_key(e["field"]) in fields]
    # Preferred source order: structured colour attribute -> structured colour column -> variant -> name tail/name.
    priorities = {norm_key("Shopify Farbe"): 0, norm_key("color_name"): 1, norm_key("ColorName"): 1,
                  norm_key("Variant"): 2, norm_key("name_tail"): 3, norm_key("name"): 4}
    contexts.sort(key=lambda et: priorities.get(norm_key(et[0]["field"]), 9))
    for e, text in contexts:
        parsed = _colour_parse(text)
        if not parsed: continue
        is_struct = e["source_type"] == "product_attribute"
        st = "product_attribute" if is_struct else "product_text"
        rel = _candidate_reliability(st)
        groups = [x for x in parsed if x[1] >= 0.2]
        if len(groups) >= 2 and "Mix" in allowed_values:
            _add_candidate(bucket, "Mix", rel, st, e["source_label"], f"Multiple colour groups in {e['source_label']}: " + "; ".join(f"{x[0]} ({x[2]})" for x in groups), allowed_values)
        for group, prob, term in parsed:
            mapped = _map_colour_group(rk, attribute, group, allowed_values, prob)
            for value, mapped_prob in mapped.items():
                _add_candidate(bucket, value, rel * (0.65 + 0.35 * mapped_prob), st, e["source_label"],
                               f"Colour term {term!r} in {e['source_label']} maps to {value!r}.", allowed_values)
        # First usable explicit colour source wins to avoid double-counting duplicate name/variant text.
        break

def _image_colour_groups(image_path):
    if not PIL_AVAILABLE: return {}
    try:
        import cv2
        im = Image.open(image_path).convert("RGB").resize((160, 160))
        rgb = np.asarray(im)
        h,w,_ = rgb.shape
        corners = np.concatenate([rgb[:8,:8].reshape(-1,3), rgb[:8,-8:].reshape(-1,3), rgb[-8:,:8].reshape(-1,3), rgb[-8:,-8:].reshape(-1,3)], axis=0)
        bg = np.median(corners.astype(float), axis=0)
        dist = np.sqrt(((rgb.astype(float)-bg)**2).sum(axis=2))
        mask = dist > 25
        # Preserve large white/neutral objects where the background subtraction is uncertain by using the central region.
        if int(mask.sum()) < 400:
            mask = np.zeros((h,w), dtype=bool); mask[int(.12*h):int(.88*h), int(.12*w):int(.88*w)] = True
        pix = rgb[mask]
        if len(pix) > COLOUR_IMAGE_SETTINGS["max_pixels"]:
            rng = np.random.default_rng(17); pix = pix[rng.choice(len(pix), COLOUR_IMAGE_SETTINGS["max_pixels"], replace=False)]
        hsv = cv2.cvtColor(pix.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1,3)
        H,S,V = hsv[:,0]/180.0, hsv[:,1]/255.0, hsv[:,2]/255.0
        groups = []
        for hh,ss,vv in zip(H,S,V):
            if vv > .82 and ss < .16: g = "Light Neutrals"
            elif vv < .20 and ss < .30: g = "Deep Shadows & Graphites"
            elif ss < .18: g = "Cool Grays & Minerals"
            elif .035 <= hh <= .15 and vv < .82 and ss < .78: g = "Warm Earth & Natural Woods"
            else: g = "Chromatic Accents & Jewel Tones"
            groups.append(g)
        counts = Counter(groups); total = max(sum(counts.values()),1)
        probs = {g: n/total for g,n in counts.items() if n/total >= COLOUR_IMAGE_SETTINGS["drop_cluster_below"]}
        # If strong chromatic + neutral/wood groups coexist, represent the dominant visual palette as Mix when allowed.
        return probs
    except Exception: return {}

def _colour_image_candidates(target, attribute, pid, allowed_values, bucket):
    rk = _category_rule_key(target)
    rows = IMAGE_INDEX.get(pid, [])
    if not rows: return
    palettes = [_image_colour_groups(r["final_path"]) for r in rows[:2]]
    if not any(palettes): return
    combined = Counter()
    for p in palettes:
        for g,v in p.items(): combined[g] += v / max(len(palettes),1)
    if not combined: return
    # Dominant category; do not force a label if the palette is dispersed.
    total = sum(combined.values()) or 1
    probs = {g: v/total for g,v in combined.items()}
    groups = [(g,p) for g,p in probs.items() if p >= 0.18]
    if len(groups) > 1 and "Mix" in allowed_values:
        _add_candidate(bucket, "Mix", _candidate_reliability("image_colour") * min(1.0, sum(p for _,p in groups)),
                       "image_colour", "product image palette", "Visual palette contains multiple groups: " + ", ".join(f"{g} {p:.0%}" for g,p in groups), allowed_values)
    for group, prob in probs.items():
        for value, mp in _map_colour_group(rk, attribute, group, allowed_values, prob).items():
            rel = _candidate_reliability("image_colour") * (0.55 + 0.45 * min(1.0, mp))
            _add_candidate(bucket, value, rel, "image_colour", "product image palette",
                           f"Dominant visible product colours map to {group} (estimated share {prob:.0%}).", allowed_values)

# ---------- dimension extraction / size classes ------------------------------------------------
def _attr_number(pid, names):
    vals, src = get_attr_values(pid, names)
    for v in vals:
        n = parse_number(v)
        if pd.notna(n) and n > 0: return float(n), src, v
    return None, None, None

def _table_dimensions(pid):
    p = _product_row(pid)
    dim = {}
    for field in ["length_cm", "depth_cm", "height_cm"]:
        x = p.get(field)
        if x is not None and pd.notna(x) and float(x) > 0: dim[field] = float(x)
    # ItemAttribute dimensions are stored in cm in this source; only use as fallback for missing DWH values.
    for key, attr in TABLE_DIMENSION_ATTRIBUTES.items():
        if key in ("length", "width"):
            value, src, raw = _attr_number(pid, [attr])
            if value and key not in dim: dim["length_cm" if key=="length" else "depth_cm"] = value
        if key == "diameter":
            value, src, raw = _attr_number(pid, [attr])
            if value: dim["diameter_cm"] = value
    # Text fallback for a named diameter or width/length in cm (e.g. '90 x 220 cm', 'Ø 120 cm').
    if not any(k in dim for k in ("length_cm", "depth_cm", "diameter_cm")):
        text = " ".join(e["value"] for e in CONTEXT_CACHE.get(pid, []) if norm_key(e["field"]) in {"name", "variant"})
        t = fold_text(text)
        dm = re.search(r"(?:ø|diam(?:eter)?\s*)\s*(\d{2,3}(?:[.,]\d+)?)\s*cm", t)
        if dm: dim["diameter_cm"] = parse_number(dm.group(1))
        nums = re.findall(r"(\d{2,3}(?:[.,]\d+)?)\s*[x×]\s*(\d{2,3}(?:[.,]\d+)?)\s*cm", t)
        if nums:
            dim["length_cm"] = max(parse_number(x) for x in nums[-1]); dim["depth_cm"] = min(parse_number(x) for x in nums[-1])
    return dim

def _seats_from_dimensions(pid):
    dim = _table_dimensions(pid)
    dia = dim.get("diameter_cm")
    if not dia and dim.get("length_cm") and dim.get("depth_cm") and abs(dim["length_cm"]-dim["depth_cm"]) <= 0.08*max(dim["length_cm"], dim["depth_cm"]):
        dia = max(dim["length_cm"], dim["depth_cm"])
    if dia:
        size, basis = dia, "round diameter"
        rules = SEAT_RULES["round"]
    elif dim.get("length_cm"):
        size = max(dim.get("length_cm",0), dim.get("depth_cm",0)); basis = "longer rectangular side"
        rules = SEAT_RULES["rectangular"]
    else: return None, None, "No reliable table dimensions found."
    prefix = next(label for edge,label in rules if size <= edge)
    allowed = _allowed_for("Esstisch", "Size (seats)")
    value = next((v for v in allowed if v.startswith(prefix + ":")), None)
    if not value: return None, None, "Dimension bin does not match the configured allowed labels."
    return value, size, f"{basis} is {size:.1f} cm; configured seat-size rule maps to {value}."

def _lounge_width(pid):
    # Use explicit product attribute in cm when present.
    val, src, raw = _attr_number(pid, LOUNGE_SEAT_WIDTH_ATTRIBUTES)
    if val and 35 <= val <= 200:
        return val, "product_attribute", f"Existing {src}={raw} cm.", None
    if not CFG["assignment"]["lounge_seat_width_estimate"]: return None, None, "No seat-width field found.", None
    # Calibrate overall-width offset from real seats with structured seat-width evidence.
    ref = []
    for other in SELECTED.loc[SELECTED["subcategory"] == "Loungesessel", "product_id"]:
        w, _, _ = _attr_number(other, LOUNGE_SEAT_WIDTH_ATTRIBUTES)
        if not w or not (35 <= w <= 200): continue
        r = _product_row(other)
        ds = [r.get("length_cm"), r.get("depth_cm")]
        ds = [float(x) for x in ds if x is not None and pd.notna(x) and float(x) > 0]
        if ds: ref.append(max(ds)-w)
    offset = float(np.median(ref)) if len(ref) >= 8 else LOUNGE_FALLBACK_OFFSET_CM
    p = _product_row(pid)
    ds = [p.get("length_cm"), p.get("depth_cm")]
    ds = [float(x) for x in ds if x is not None and pd.notna(x) and float(x) > 0]
    if not ds: return None, None, "No overall width to estimate from.", offset
    est = max(ds)-offset
    if not 35 <= est <= 200: return None, None, "Estimated seat width was outside the plausible range.", offset
    return est, "estimate", f"Estimated seat width {est:.1f} cm = overall chair width {max(ds):.1f} cm minus median seat/frame offset {offset:.1f} cm; low-confidence fallback.", offset

def _size_class(value, edges, allowed_values, less_first=False):
    if value is None or not pd.notna(value): return None
    if less_first:
        label = "XS" if value < edges[0][0] else next((lab for edge,lab in edges if value <= edge), "L")
    else: label = next((lab for edge,lab in edges if value <= edge), edges[-1][1])
    return next((v for v in allowed_values if v.startswith(label + ":")), None)

def _dimension_candidates(target, attribute, pid, allowed_values, bucket):
    if target == "Esstisch" and attribute == "Size (seats)":
        value, size, explanation = _seats_from_dimensions(pid)
        if value:
            # Slightly reduce confidence near a threshold; seating is estimated from dimensions, not directly specified.
            rel = _candidate_reliability("dimensions")
            if size is not None:
                all_edges = [x[0] for x in (SEAT_RULES["round"] if _table_dimensions(pid).get("diameter_cm") else SEAT_RULES["rectangular"])[:-1]]
                if any(abs(size-e) <= max(3, e*SEAT_RULES["softness"]) for e in all_edges): rel *= 0.78
            _add_candidate(bucket, value, rel, "dimensions", "product dimensions", explanation, allowed_values)
    elif target == "Loungesessel" and attribute == "Size (seat width)":
        width, kind, explanation, offset = _lounge_width(pid)
        if width:
            value = _size_class(width, LOUNGE_SEAT_WIDTH_EDGES, allowed_values, less_first=True)
            if value:
                source = "dimensions" if kind == "product_attribute" else "estimate"
                rel = _candidate_reliability(source)
                _add_candidate(bucket, value, rel, source, "seat-width field" if kind=="product_attribute" else "calibrated width estimate", explanation, allowed_values)
    elif target == "Vase" and attribute == "Form & flower fit":
        p = _product_row(pid); d = _table_dimensions(pid)
        h = p.get("height_cm"); h = float(h) if h is not None and pd.notna(h) else None
        dia = d.get("diameter_cm")
        if dia is None:
            vals = [p.get("length_cm"), p.get("depth_cm")]
            vals = [float(x) for x in vals if x is not None and pd.notna(x) and float(x)>0]
            dia = max(vals) if vals else None
        txt = " ".join(t for e,t in _folded_context(pid) if norm_key(e["field"]) in {"name", "variant"})
        value, expl = None, ""
        if re.search(r"sculptur|deko object|kunstobjekt|object vase|art object|vase sculpture", txt):
            value, expl = "Sculptural/Objects", "Product name explicitly describes a sculptural/decorative object."
        elif h and h >= VASE_RULES["floor_height_cm"]:
            value, expl = "Floor vase", f"Vase height {h:.1f} cm is at/above the configured {VASE_RULES['floor_height_cm']} cm floor-vase threshold."
        elif re.search(r"tub|tube|solifleur|bud vase|single stem|flower tube|vasenrohr", txt) and (dia is None or dia <= VASE_RULES["tube_min_diameter_cm"]):
            value, expl = "Tubes", "Name indicates a tube/solifleur/bud vase and dimensions do not contradict a narrow tube."
        elif h and dia and dia > 0:
            ratio = h/dia
            if ratio >= VASE_RULES["slender_ratio"] and dia <= VASE_RULES["slender_max_diameter_cm"]:
                value, expl = "Slender", f"Height/diameter ratio {ratio:.2f} and diameter {dia:.1f} cm fit the slender-vase rule."
            elif ratio <= VASE_RULES["round_max_ratio"]:
                value, expl = "Round", f"Height/diameter ratio {ratio:.2f} fits the round/bellied-vase rule."
        if value in allowed_values:
            _add_candidate(bucket, value, _candidate_reliability("dimensions"), "dimensions", "product dimensions/name form",
                           expl, allowed_values)

# ---------- image and PDF similarity candidates ------------------------------------------------
def _image_candidates(target, attribute, pid, allowed_values, bucket):
    rk = _category_rule_key(target)
    spec = IMAGE_PROMPTS.get(rk, {}).get(attribute)
    if spec:
        probs, meta = _image_prompt_scores(pid, rk, attribute, allowed_values, spec)
        if probs:
            multi = attribute in CFG["assignment"]["multi_value_attributes"].get(rk, [])
            selected = [(v,p) for v,p in probs.items() if p >= CFG["assignment"]["multi_value_threshold"]] if multi else sorted(probs.items(), key=lambda x: -x[1])[:1]
            for value, prob in selected:
                if value not in allowed_values: continue
                factor = float(spec.get("reliability_factor", 1.0))
                rel = _candidate_reliability("image") * factor * (0.55 + 0.45 * float(prob))
                _add_candidate(bucket, value, rel, "image", "product image embedding",
                               f"Cropped product-photo embedding best matches the visual description for {value!r} (model {meta.get('model')}, normalized score {prob:.2f}).",
                               allowed_values, meta)
    if attribute in ("Colour Group", "Colour & finish"):
        _colour_image_candidates(target, attribute, pid, allowed_values, bucket)

# ---------- style candidates: PDF examples + text/image prompt ---------------------------------
def _style_candidates(target, pid, allowed_values, bucket):
    rk = _category_rule_key(target)
    visual = PRODUCT_VISUAL.get(pid)
    probs, meta = _style_probabilities(rk, visual) if visual is not None else ({}, {"status": "no product image"})
    if probs:
        style = max(probs, key=probs.get)
        prob = float(probs[style])
        has_own = bool(meta.get("has_own_examples"))
        loo = STYLE_RELIABILITY.get(rk, 0.50) if has_own else CFG["pdf"]["transfer_reliability_factor"]
        similarity = float(meta.get("nearest_similarity", 0.0))
        near = similarity >= CFG["pdf"]["near_duplicate_similarity"]["deep"] and meta.get("nearest_style") == style
        if has_own:
            rel = 0.30 + 0.45*loo + 0.15*max(0.0, min(1.0, similarity)) + 0.10*prob
            if near: rel = max(rel, 0.88)
            src_label = "PDF example similarity"
            evidence = f"Nearest specialist-labelled PDF product is {similarity:.2f} cosine-similar (page {meta.get('nearest_pdf_page')}); predicted style {style} (LOO similarity reliability {loo:.2f})."
        else:
            rel = 0.33 + 0.10*prob
            src_label = "transferred PDF example similarity"
            evidence = f"No examples for this sub-category; weak transferred similarity to PDF style boards {meta.get('source_categories')} (page {meta.get('nearest_pdf_page')}); predicted {style}."
        _add_candidate(bucket, style, rel, "pdf_example", src_label, evidence, allowed_values, meta)
    # Product-image/text joint model style prompts; separate from the labeled-example retrieval.
    prompt_probs, pmeta = _image_prompt_scores(pid, rk, "Design style", allowed_values)
    if prompt_probs:
        style = max(prompt_probs, key=prompt_probs.get); prob = float(prompt_probs[style])
        rel = _candidate_reliability("style_zeroshot") * (0.65 + 0.35*prob)
        _add_candidate(bucket, style, rel, "image", "product image embedding (style description)",
                       f"Cropped product-photo embedding best matches the {style} specialist-style description (score {prob:.2f}).", allowed_values, pmeta)
    # Literal style hints are deliberately weak; use them only as a supporting signal.
    text_fields = {norm_key(f) for f in STYLE_TEXT_FIELDS}
    for e, text in _folded_context(pid):
        if norm_key(e["field"]) not in text_fields: continue
        for style, patterns in STYLE_TEXT_HINTS.items():
            style_value = _exact_allowed(style, allowed_values)
            if not style_value: continue
            if any(re.search(pat, text, re.I) for pat in patterns):
                src = "product_attribute" if e["source_type"] == "product_attribute" else "product_text"
                _add_candidate(bucket, style_value, _candidate_reliability("style_text_hint"), src, e["source_label"],
                               f"Weak literal style cue in {e['source_label']} matches {style_value!r}.", allowed_values)

# ---------- generic text fallback for new attributes -------------------------------------------
def _generic_text_candidates(target, attribute, pid, allowed_values, bucket):
    # Skip vague one-word labels such as 'Other', 'M', 'Set' to avoid accidental matches.
    for e, text in _folded_context(pid):
        for value in allowed_values:
            if norm_key(value) in {"other", "mix", "set", "none"} or len(norm_key(value)) < 5: continue
            pat = re.escape(fold_text(value))
            if re.search(r"(?<!\w)" + pat + r"(?!\w)", text, re.I):
                src = "product_attribute" if e["source_type"] == "product_attribute" else "product_text"
                _add_candidate(bucket, value, _candidate_reliability("generic_text"), src, e["source_label"],
                               f"Allowed value {value!r} appears literally in {e['source_label']}.", allowed_values)

# ---------- confidence aggregation -------------------------------------------------------------
def _aggregate_candidates(bucket, allowed_values):
    scores, evidence_by_value = {}, {}
    for val in allowed_values:
        rows = bucket.get(val, [])
        by_type = {}
        for row in rows:
            t = row["source_type"]
            if t not in by_type or row["reliability"] > by_type[t]["reliability"]: by_type[t] = row
        reliability = [max(0.0, min(.99, r["reliability"])) for r in by_type.values()]
        score = 1.0 - np.prod([1.0-x for x in reliability]) if reliability else 0.0
        scores[val] = float(score)
        evidence_by_value[val] = list(by_type.values())
    ranked = sorted(scores.items(), key=lambda x: (-x[1], allowed_values.index(x[0])))
    if not ranked or ranked[0][1] <= 0: return None, 0.0, [], scores
    best, best_raw = ranked[0]
    runner = ranked[1][1] if len(ranked)>1 else 0.0
    adjusted = max(0.0, best_raw - CFG["assignment"]["conflict_penalty"]*runner)
    return best, float(adjusted), evidence_by_value.get(best, []), scores

def _confidence_level(score):
    levels = CFG["assignment"]["levels"]
    if score >= levels["High"]: return "High"
    if score >= levels["Medium"]: return "Medium"
    if score >= levels["Low"]: return "Low"
    return "Not determinable"

def _source_phrase(rows):
    types = {r["source_type"] for r in rows}
    if len(types) > 1: return "combined evidence (" + ", ".join(sorted(types)) + ")"
    if not types: return "no reliable evidence"
    return {"product_attribute": "product attribute", "product_text": "product text", "dimensions": "dimensions",
            "estimate": "dimension-based estimate", "image": "image embedding", "image_colour": "product image",
            "pdf_example": "PDF example similarity"}.get(next(iter(types)), next(iter(types)))

def _assignment_for(target, pid, attribute):
    allowed_values = _allowed_for(target, attribute)
    if not allowed_values:
        return {"assigned_value": NOT_DETERMINABLE, "confidence": "Not determinable", "confidence_score": 0.0,
                "evidence": "No allowed values found for this attribute/category.", "evidence_source": "no reliable evidence", "method": "no_allowed_values", "is_sentinel": True}
    bucket = defaultdict(list)
    # 1. Direct configured mappings from your optional new columns.
    _column_mapping_candidates(target, attribute, pid, allowed_values, bucket)
    # 2. Exact structured product attribute (never force a non-exact conversion here).
    _direct_attribute_candidates(target, attribute, pid, allowed_values, bucket)
    # 3. Attribute-specific rules / styles / colors / dimensions.
    providers = PIPELINE.get(_category_rule_key(target), {}).get(attribute, DEFAULT_PIPELINE)
    _ = providers  # exposed in the notebook; the explicit dispatch below makes provider order auditable.
    if attribute == "Design style": _style_candidates(target, pid, allowed_values, bucket)
    _keyword_candidates(target, attribute, pid, allowed_values, bucket)
    if attribute in ("Colour Group", "Colour & finish"):
        _colour_text_candidates(target, attribute, pid, allowed_values, bucket)
    _dimension_candidates(target, attribute, pid, allowed_values, bucket)
    if attribute in IMAGE_PROMPTS.get(_category_rule_key(target), {}) or attribute in ("Colour Group", "Colour & finish"):
        _image_candidates(target, attribute, pid, allowed_values, bucket)
    _generic_text_candidates(target, attribute, pid, allowed_values, bucket)

    # Attribute-level configured absence defaults are explicit assumptions (not silent guesses).
    if not any(bucket.values()):
        default = ABSENCE_DEFAULTS.get(_category_rule_key(target), {}).get(attribute)
        if default:
            val, rel = default
            if val in allowed_values:
                _add_candidate(bucket, val, rel, "estimate", "configured absence default",
                               f"Configured default: no configured text/attribute evidence of an alternative was found for {attribute!r}; assumed {val!r}.", allowed_values)

    multi = attribute in CFG["assignment"]["multi_value_attributes"].get(_category_rule_key(target), [])
    if multi:
        values = []
        per_value = []
        for value in allowed_values:
            rows = bucket.get(value, [])
            if not rows: continue
            # Per-value score uses best evidence by source type.
            score = 1 - np.prod([1-r for r in [max(x["reliability"] for x in rows if x["source_type"]==t) for t in {x["source_type"] for x in rows}]])
            if score >= CFG["assignment"]["multi_value_threshold"]:
                values.append(value); per_value.append((value, float(score), rows))
        if values:
            allrows = [r for _,_,rows in per_value for r in rows]
            score = min(s for _,s,_ in per_value)
            level = _confidence_level(score)
            evidence = " | ".join(dict.fromkeys(r["evidence"] for r in allrows))
            return {"assigned_value": CFG["assignment"]["multi_value_separator"].join(values), "confidence": level,
                    "confidence_score": score, "evidence": evidence, "evidence_source": _source_phrase(allrows),
                    "method": ";".join(sorted({r["source_type"] for r in allrows})), "is_sentinel": False}
        return {"assigned_value": NOT_DETERMINABLE, "confidence": "Not determinable", "confidence_score": 0.0,
                "evidence": "No add-on value can be confirmed from the available product fields or images.",
                "evidence_source": "no reliable evidence", "method": "no_evidence", "is_sentinel": True}

    best, score, rows, scores = _aggregate_candidates(bucket, allowed_values)
    if best is None or score < CFG["assignment"]["levels"]["Low"]:
        return {"assigned_value": NOT_DETERMINABLE, "confidence": "Not determinable", "confidence_score": score,
                "evidence": "Insufficient reliable evidence to select a valid category value.",
                "evidence_source": "no reliable evidence", "method": "insufficient_evidence", "is_sentinel": True}
    level = _confidence_level(score)
    # Style is subjective: only a near-duplicate to an example may be High; no own PDF examples -> Low maximum.
    if attribute == "Design style":
        rk = _category_rule_key(target)
        sm = _style_probabilities(rk, PRODUCT_VISUAL.get(pid))[1] if PRODUCT_VISUAL.get(pid) is not None else {}
        own_label_examples = bool(len(PDF_EXAMPLES) and ((PDF_EXAMPLES["rule_key"]==rk) & (PDF_EXAMPLES["style_value"]==best)).any())
        has_own = bool(sm.get("has_own_examples"))
        near = sm.get("nearest_style") == best and float(sm.get("nearest_similarity", 0.0)) >= CFG["pdf"]["near_duplicate_similarity"]["deep"]
        if not has_own: level = "Low" if level in ("High", "Medium") else level
        elif not own_label_examples: level = "Low" if level in ("High", "Medium") else level
        elif not near and level == "High": level = "Medium"
    evidence = " | ".join(dict.fromkeys(r["evidence"] for r in rows)) if rows else ""
    return {"assigned_value": best, "confidence": level, "confidence_score": score,
            "evidence": evidence[:1500], "evidence_source": _source_phrase(rows),
            "method": ";".join(sorted({r["source_type"] for r in rows})), "is_sentinel": False}

# ---------- build exactly one product-attribute row for every required pair --------------------
_assignment_rows = []
for i, p in SELECTED.iterrows():
    target = p["subcategory"]
    for attribute, allowed_values in ALLOWED[MATRIX_CATEGORY[target]].items():
        result = _assignment_for(target, p["product_id"], attribute)
        chosen = result["assigned_value"]
        is_multi = attribute in CFG["assignment"]["multi_value_attributes"].get(_category_rule_key(target), [])
        if not result["is_sentinel"]:
            parts = chosen.split(CFG["assignment"]["multi_value_separator"]) if is_multi else [chosen]
            invalid = [v for v in parts if v not in allowed_values]
            if invalid:
                dq("assignment", "invalid assigned value corrected to Not determinable", len(invalid), "error", f"{target}/{attribute}: {invalid}")
                result.update({"assigned_value": NOT_DETERMINABLE, "confidence": "Not determinable", "confidence_score": 0.0,
                               "evidence": "Candidate value was not in the category's allowed list.", "evidence_source": "no reliable evidence",
                               "method": "invalid_candidate", "is_sentinel": True})
        _assignment_rows.append({"product_id": p["product_id"], "product_name": p["product_name"], "subcategory": target,
                                 "matrix_category": MATRIX_CATEGORY[target], "attribute_name": attribute,
                                 "assigned_value": result["assigned_value"], "confidence": result["confidence"],
                                 "confidence_score": round(float(result["confidence_score"]), 4), "evidence": result["evidence"],
                                 "evidence_source": result["evidence_source"], "method": result["method"],
                                 "allowed_values": " | ".join(allowed_values), "is_sentinel": bool(result["is_sentinel"]),
                                 "image_count": len(PRODUCT_IMAGE_PATHS.get(p["product_id"], [])),
                                 "model_backend": MODEL_BACKEND, "model_name": ENCODER.name})

ASSIGNMENTS = pd.DataFrame(_assignment_rows)
log(f"attribute assignments: {len(ASSIGNMENTS):,} rows for {ASSIGNMENTS['product_id'].nunique():,} products")

# Assignment audit tables.
ASSIGNMENT_SUMMARY = (ASSIGNMENTS.assign(completed=~ASSIGNMENTS["is_sentinel"])
                      .groupby(["subcategory", "attribute_name"], sort=False)
                      .agg(required=("assigned_value", "size"), completed=("completed", "sum"),
                           not_determinable=("is_sentinel", "sum"),
                           completed_pct=("completed", lambda x: 100*x.mean()))
                      .reset_index())
CONFIDENCE_SUMMARY = (ASSIGNMENTS.groupby(["subcategory", "confidence"], sort=False).size().rename("assignments").reset_index())
UNCERTAIN_ASSIGNMENTS = ASSIGNMENTS[(ASSIGNMENTS["confidence"].isin(["Low", "Not determinable"])) | ASSIGNMENTS["is_sentinel"]].copy()

display(ASSIGNMENT_SUMMARY)
display(CONFIDENCE_SUMMARY)
display(UNCERTAIN_ASSIGNMENTS[["product_id", "product_name", "subcategory", "attribute_name", "assigned_value", "confidence", "evidence_source"]].head(30))

# %% [markdown]
# ## 10. Sales loading, date-period limitation and aggregation
# The configured period is **2025-11-01 through 2026-10-31 inclusive**. The supplied sales file is a product-level aggregate
# with net/gross intake and quantity fields but **no date or order identifier**. Therefore it cannot be filtered to that period;
# period metrics stay missing (`NaN`, not zero). Available all-time aggregate figures are included separately as clearly labelled
# context only. The file's currency is not identified, so the configured EUR label is an *unverified assumption*, not a source fact.
# Sales row count is not misrepresented as orders; order count and first/last sale date remain missing.

# %%
SALES_PERIOD_START = pd.Timestamp(CFG["sales"]["period_start"])
SALES_PERIOD_END = pd.Timestamp(CFG["sales"]["period_end"])
SALES_PERIOD_STATUS = "unavailable"
SALES_DATE_COLUMN = None
SALES_PRODUCT_KEY = None
SALES_MEASURES = {}
SALES_CURRENCY = CFG["sales"]["currency"]
SALES_CURRENCY_VERIFIED = CFG["sales"]["currency_verified"]
SALES_SUMMARY = pd.DataFrame()
SALES_SOURCE_CLEAN = pd.DataFrame()
SALES_PERIOD_REPORT = {}

if "sales" in SOURCES:
    raw = SOURCES["sales"].copy()
    sm = FOUND["sales"]
    SALES_PRODUCT_KEY = sm.get("product_id")
    SALES_DATE_COLUMN = sm.get("date")
    logical_fields = ["units_gross", "revenue_gross", "units_net", "revenue_net", "stock_available"]
    s = pd.DataFrame(index=raw.index)
    s["product_id"] = raw[SALES_PRODUCT_KEY].map(clean_id) if SALES_PRODUCT_KEY else np.nan
    s["sales_name"] = raw[sm["product_name"]].map(clean_text) if sm.get("product_name") else np.nan
    s["item_type"] = raw[sm["item_type"]].map(clean_text) if sm.get("item_type") else np.nan
    for logical in logical_fields:
        col = sm.get(logical)
        s[logical] = raw[col].map(parse_number) if col else np.nan
        if col: SALES_MEASURES[logical] = col
    if SALES_DATE_COLUMN:
        s["sale_date"] = pd.to_datetime(raw[SALES_DATE_COLUMN], errors="coerce", dayfirst=CFG["sales"]["dayfirst"])
        date_min = s["sale_date"].min(); date_max = s["sale_date"].max()
        date_min = date_min.date().isoformat() if pd.notna(date_min) else None
        date_max = date_max.date().isoformat() if pd.notna(date_max) else None
    else:
        s["sale_date"] = pd.NaT
        date_min = date_max = None
    if sm.get("order_id"):
        s["order_id"] = raw[sm["order_id"]].map(clean_text)
    else: s["order_id"] = np.nan

    # Remove summary/total rows, keep only numeric product keys; selected IDs are matched by the same product key, not name.
    total_mask = s["product_id"].map(lambda x: isinstance(x, str) and norm_key(x) in {"total", "grandtotal", "gesamt"})
    numeric_id_mask = s["product_id"].map(lambda x: isinstance(x, str) and bool(re.fullmatch(r"-?\d+", x)))
    removed_total = int(total_mask.sum())
    removed_bad_key = int((~numeric_id_mask & ~total_mask).sum())
    s = s[numeric_id_mask & ~total_mask].copy()
    dq("sales", "summary/Total rows excluded", removed_total, "info")
    dq("sales", "sales rows without numeric product key excluded", removed_bad_key, "warning" if removed_bad_key else "info")
    # Date availability determines whether period metrics may be produced.
    if SALES_DATE_COLUMN and s["sale_date"].notna().any():
        in_period = s["sale_date"].between(SALES_PERIOD_START, SALES_PERIOD_END, inclusive="both")
        SALES_PERIOD_STATUS = "date_column_found_period_filtered"
        period_rows = s[in_period].copy()
        date_min = s["sale_date"].min().date().isoformat()
        date_max = s["sale_date"].max().date().isoformat()
    else:
        period_rows = s.iloc[:0].copy()
        SALES_PERIOD_STATUS = "unavailable_no_date_column" if not SALES_DATE_COLUMN else "unavailable_no_valid_dates"
        date_min = date_max = None
    SALES_SOURCE_CLEAN = s.copy()

    # All-time aggregate: supplemental context only. If duplicate product IDs occur, sum measures once per source row.
    unit_col = CFG["sales"]["primary_units"]
    rev_col = CFG["sales"]["primary_revenue"]
    group_cols = ["product_id"]
    agg_spec = {}
    for logical in ["units_gross", "revenue_gross", "units_net", "revenue_net", "stock_available"]:
        if logical in s.columns and s[logical].notna().any(): agg_spec[logical] = "sum"
    alltime = s.groupby(group_cols, dropna=True).agg(agg_spec).reset_index() if agg_spec else pd.DataFrame(columns=["product_id"])
    dup_sales_keys = int(s["product_id"].duplicated(keep=False).sum())
    dq("sales", "source sales rows sharing a product key (aggregated by product)", dup_sales_keys, "warning" if dup_sales_keys else "info")
    alltime["sales_rows_all_time"] = s.groupby("product_id").size().reindex(alltime["product_id"]).to_numpy() if len(alltime) else []
    # Order count only when a true order identifier exists; do not infer it from units.
    if sm.get("order_id"):
        order_counts = s.dropna(subset=["order_id"]).groupby("product_id")["order_id"].nunique().rename("orders_all_time").reset_index()
        alltime = alltime.merge(order_counts, on="product_id", how="left", validate="one_to_one")
    else:
        alltime["orders_all_time"] = np.nan
    # Period aggregate is empty when no dates are provided; when date exists, aggregate only within requested period.
    if len(period_rows):
        period_agg = {}
        for logical in [unit_col, rev_col]:
            if logical in period_rows and period_rows[logical].notna().any(): period_agg[logical] = "sum"
        if sm.get("order_id"): period_agg["order_id"] = "nunique"
        period = period_rows.groupby("product_id").agg(period_agg).reset_index()
        if sm.get("order_id"): period = period.rename(columns={"order_id": "orders_sold_period"})
        if sm.get("order_id") is None: period["orders_sold_period"] = np.nan
        if period_rows["sale_date"].notna().any():
            dates = period_rows.groupby("product_id")["sale_date"].agg(first_sale_date="min", last_sale_date="max").reset_index()
            period = period.merge(dates, on="product_id", validate="one_to_one")
    else:
        period = pd.DataFrame(columns=["product_id", unit_col, rev_col, "orders_sold_period", "first_sale_date", "last_sale_date"])

    # Rename source measures into explicit unit names; retain exact source header in the report.
    alltime = alltime.rename(columns={"units_gross": "units_gross_all_time", "revenue_gross": "revenue_gross_all_time",
                                      "units_net": "units_net_all_time", "revenue_net": "revenue_net_all_time",
                                      "stock_available": "stock_available_source_snapshot"})
    period_rename = {unit_col: "units_sold_requested_period", rev_col: "revenue_requested_period"}
    period = period.rename(columns=period_rename)
    SALES_SUMMARY = alltime.merge(period, on="product_id", how="left", validate="one_to_one")
    # No matched sales row is missing, never silently converted to zero.
    matched_keys = set(SALES_SUMMARY["product_id"])
    selected_sales = SELECTED[["product_id", "subcategory"]].merge(SALES_SUMMARY, on="product_id", how="left", validate="one_to_one")
    selected_sales["sales_row_present"] = selected_sales["product_id"].isin(matched_keys)
    selected_sales["sales_currency_configured"] = SALES_CURRENCY
    selected_sales["sales_currency_verified"] = bool(SALES_CURRENCY_VERIFIED)
    selected_sales["sales_period_status"] = SALES_PERIOD_STATUS
    # Only an observed source row with a numeric zero is a zero sale; unmatched product is NaN.
    rank_col = "units_sold_requested_period" if SALES_PERIOD_STATUS == "date_column_found_period_filtered" else None
    if rank_col and rank_col in selected_sales:
        selected_sales["sales_rank_requested_period"] = selected_sales.groupby("subcategory")[rank_col].rank(method="min", ascending=False)
    else:
        selected_sales["sales_rank_requested_period"] = np.nan
    # Separate, explicitly all-time proxy ranking for the supplied aggregate file.
    proxy_units = {"units_net": "units_net_all_time", "units_gross": "units_gross_all_time"}.get(unit_col, "units_net_all_time")
    if proxy_units in selected_sales:
        selected_sales["sales_rank_all_time_proxy"] = selected_sales.groupby("subcategory")[proxy_units].rank(method="min", ascending=False)
    else: selected_sales["sales_rank_all_time_proxy"] = np.nan
    if SALES_PERIOD_STATUS == "unavailable_no_date_column":
        selected_sales["units_sold_requested_period"] = np.nan
        selected_sales["revenue_requested_period"] = np.nan
        selected_sales["orders_sold_period"] = np.nan
        selected_sales["first_sale_date"] = pd.NaT
        selected_sales["last_sale_date"] = pd.NaT
    # At most one sales row per product is allowed after aggregation.
    assert not SALES_SUMMARY["product_id"].duplicated().any(), "Sales summary is not unique by product key"
    assert len(selected_sales) == len(SELECTED), "Sales join changed selected product row count"

    SALES_PERIOD_REPORT = {
        "requested_period_start": SALES_PERIOD_START.date().isoformat(), "requested_period_end_inclusive": SALES_PERIOD_END.date().isoformat(),
        "period_status": SALES_PERIOD_STATUS, "date_column": SALES_DATE_COLUMN, "available_date_min": date_min, "available_date_max": date_max,
        "product_key_column": SALES_PRODUCT_KEY, "sales_measures": SALES_MEASURES, "primary_units_measure": unit_col,
        "primary_revenue_measure": rev_col, "currency": SALES_CURRENCY, "currency_verified_from_source": SALES_CURRENCY_VERIFIED,
        "filters": {"excluded_total_rows": removed_total, "excluded_non_numeric_product_key_rows": removed_bad_key,
                    "item_type_filter": "none (all numeric product-key rows retained)",
                    "period_filter": "none possible: no date field" if not SALES_DATE_COLUMN else "inclusive date range"},
        "aggregation_rules": {"grain": "product_id", "duplicate_product_rows": "sum available quantity/revenue measures; count unique order IDs only when present",
                              "missing_sales": "NaN/missing, not zero", "order_count": "not inferred from item quantity",
                              "first_last_sale_dates": "only when sale-date column and dates are present"},
        "limitations": []}
    if SALES_PERIOD_STATUS != "date_column_found_period_filtered":
        SALES_PERIOD_REPORT["limitations"].append("Source file has no usable date column. Requested period sales are unavailable; period fields remain missing.")
    if not sm.get("order_id"):
        SALES_PERIOD_REPORT["limitations"].append("No order ID column; order counts are unavailable (not inferred from units).")
    if not SALES_CURRENCY_VERIFIED:
        SALES_PERIOD_REPORT["limitations"].append(f"Currency is not identified in the source; {SALES_CURRENCY} is an unverified configuration assumption.")

    dq("sales", "date column available", int(bool(SALES_DATE_COLUMN)), "info" if SALES_DATE_COLUMN else "warning",
       f"period status={SALES_PERIOD_STATUS}; range={date_min}..{date_max}")
    dq("sales", "selected products with matching sales product key", int(selected_sales["sales_row_present"].sum()), "info")
    dq("sales", "selected products with no sales row (missing, not zero)", int((~selected_sales["sales_row_present"]).sum()), "warning" if (~selected_sales["sales_row_present"]).any() else "info")
    if SALES_PERIOD_STATUS != "date_column_found_period_filtered":
        dq("sales", "requested-period metrics unavailable", len(SELECTED), "warning", "No date column; no requested-period top sellers/rank can be calculated.")
    else:
        # Support top sellers by requested period when source date data exists.
        sort_metric = "units_sold_requested_period" if CFG["sales"]["rank_by"] == "units" else "revenue_requested_period"
        top_period = selected_sales.sort_values(sort_metric, ascending=False, na_position="last").groupby("subcategory", sort=False).head(10)
        display(top_period[["product_id", "subcategory", sort_metric, "sales_rank_requested_period"]])
    # all-time proxy top list, kept separate so it is never mistaken for the requested period
    if proxy_units in selected_sales:
        TOP_SELLERS_ALL_TIME_PROXY = (selected_sales.sort_values(proxy_units, ascending=False, na_position="last")
                                      .groupby("subcategory", sort=False).head(10)
                                      [["product_id", "subcategory", "product_id", proxy_units, "sales_rank_all_time_proxy"]])
    else: TOP_SELLERS_ALL_TIME_PROXY = pd.DataFrame()
    SALES_FOR_PRODUCTS = selected_sales
    log(f"sales: {SALES_PERIOD_STATUS}; date column={SALES_DATE_COLUMN}; product key={SALES_PRODUCT_KEY}; all-time matched {selected_sales['sales_row_present'].sum():,}/{len(SELECTED):,}")
else:
    SALES_PERIOD_STATUS = "sales_file_missing"
    SALES_PERIOD_REPORT = {"period_status": SALES_PERIOD_STATUS, "requested_period_start": SALES_PERIOD_START.date().isoformat(),
                          "requested_period_end_inclusive": SALES_PERIOD_END.date().isoformat(), "limitations": ["Sales source file was missing."]}
    SALES_FOR_PRODUCTS = SELECTED[["product_id", "subcategory"]].copy()
    SALES_FOR_PRODUCTS["sales_row_present"] = False
    SALES_FOR_PRODUCTS["sales_currency_configured"] = SALES_CURRENCY
    SALES_FOR_PRODUCTS["sales_currency_verified"] = bool(SALES_CURRENCY_VERIFIED)
    SALES_FOR_PRODUCTS["sales_period_status"] = SALES_PERIOD_STATUS
    for c in ["units_sold_requested_period", "revenue_requested_period", "orders_sold_period", "first_sale_date", "last_sale_date",
              "sales_rank_requested_period", "sales_rows_all_time", "units_net_all_time", "revenue_net_all_time",
              "units_gross_all_time", "revenue_gross_all_time", "orders_all_time", "sales_rank_all_time_proxy"]:
        SALES_FOR_PRODUCTS[c] = np.nan
    TOP_SELLERS_ALL_TIME_PROXY = pd.DataFrame()
    dq("sales", "sales source file missing", 1, "warning")

display(pd.DataFrame([SALES_PERIOD_REPORT]))
if "SALES_FOR_PRODUCTS" in globals(): display(SALES_FOR_PRODUCTS.head(8))

# %% [markdown]
# ## 11. Validation checks
# Structural checks fail clearly (or make a warning in the report when `CFG['validation']['strict']=False`).
# The sentinel `Not determinable` is explicitly allowed as a *documented missing-evidence status*, not as a category value.
# Multi-value assignments are validated option-by-option against the relevant allowed list.

# %%
VALIDATION_ROWS = []
def validation(check, passed, observed, expected="", severity="error", detail=""):
    VALIDATION_ROWS.append({"check": check, "passed": bool(passed), "observed": str(observed), "expected": str(expected),
                            "severity": severity if passed else severity, "detail": detail})

def _v_failures(): return [r for r in VALIDATION_ROWS if not r["passed"] and r["severity"] == "error"]

# Check unique source product keys / selected keys / long assignment keys.
validation("selected product key is unique", not SELECTED["product_id"].duplicated().any(),
           int(SELECTED["product_id"].duplicated().sum()), 0, detail="One row per selected product id.")
validation("assignment product_id + attribute_name unique", not ASSIGNMENTS.duplicated(["product_id", "attribute_name"]).any(),
           int(ASSIGNMENTS.duplicated(["product_id", "attribute_name"]).sum()), 0)

# Expected product-attribute pairs from the allowed-value matrix.
_expected_pairs = set()
for _, pr in SELECTED[["product_id", "subcategory"]].iterrows():
    for attr in ALLOWED[MATRIX_CATEGORY[pr["subcategory"]]]:
        _expected_pairs.add((pr["product_id"], attr))
_observed_pairs = set(zip(ASSIGNMENTS["product_id"], ASSIGNMENTS["attribute_name"]))
_missing_pairs = _expected_pairs - _observed_pairs
_extra_pairs = _observed_pairs - _expected_pairs
validation("all required product-attribute pairs present", not _missing_pairs,
           len(_missing_pairs), 0, detail="; ".join(map(str, list(_missing_pairs)[:10])))
validation("no assignments for non-required pairs", not _extra_pairs,
           len(_extra_pairs), 0, detail="; ".join(map(str, list(_extra_pairs)[:10])))
validation("assignment row count equals expected pairs", len(ASSIGNMENTS) == len(_expected_pairs),
           len(ASSIGNMENTS), len(_expected_pairs))

# Allowed value validation. Sentinels may only be the explicitly configured value.
_invalid_values = []
for _, r in ASSIGNMENTS.iterrows():
    if bool(r["is_sentinel"]):
        if r["assigned_value"] != NOT_DETERMINABLE:
            _invalid_values.append((r["product_id"], r["attribute_name"], r["assigned_value"], "unknown sentinel"))
        continue
    allowed = _allowed_for(r["subcategory"], r["attribute_name"])
    multi = r["attribute_name"] in CFG["assignment"]["multi_value_attributes"].get(_category_rule_key(r["subcategory"]), [])
    values = r["assigned_value"].split(CFG["assignment"]["multi_value_separator"]) if multi else [r["assigned_value"]]
    for value in values:
        if value not in allowed: _invalid_values.append((r["product_id"], r["attribute_name"], value, "not in allowed list"))
validation("all non-sentinel assigned values valid for their sub-category", len(_invalid_values) == 0,
           len(_invalid_values), 0, detail="; ".join(map(str, _invalid_values[:8])))

# Non-empty fields, sentinel policy and confidence labels.
validation("no assignment values empty", ASSIGNMENTS["assigned_value"].notna().all() and ASSIGNMENTS["assigned_value"].astype(str).str.strip().ne("").all(),
           int(ASSIGNMENTS["assigned_value"].isna().sum() + ASSIGNMENTS["assigned_value"].astype(str).str.strip().eq("").sum()), 0)
valid_conf = {"High", "Medium", "Low", "Not determinable"}
validation("confidence labels use configured vocabulary", set(ASSIGNMENTS["confidence"]).issubset(valid_conf),
           sorted(set(ASSIGNMENTS["confidence"])), sorted(valid_conf))
validation("Not determinable is marked as sentinel", bool((ASSIGNMENTS.loc[ASSIGNMENTS["assigned_value"] == NOT_DETERMINABLE, "is_sentinel"]).all()),
           int((ASSIGNMENTS["assigned_value"] == NOT_DETERMINABLE).sum()), "all marked")

# No join multiplication: sales grouping and product join cardinality.
if "SALES_FOR_PRODUCTS" in globals():
    validation("sales summary is unique by product key", not SALES_SUMMARY.get("product_id", pd.Series(dtype=str)).duplicated().any(),
               int(SALES_SUMMARY.get("product_id", pd.Series(dtype=str)).duplicated().sum()), 0)
    validation("sales join did not change product row count", len(SALES_FOR_PRODUCTS) == len(SELECTED), len(SALES_FOR_PRODUCTS), len(SELECTED))
    if SALES_PERIOD_STATUS != "date_column_found_period_filtered":
        period_nonnull = int(SALES_FOR_PRODUCTS.get("units_sold_requested_period", pd.Series(dtype=float)).notna().sum())
        validation("no period sales invented without usable source dates", period_nonnull == 0, period_nonnull, 0,
                   severity="error", detail=f"period status={SALES_PERIOD_STATUS}; missing period stays NaN")
    for metric in ["units_net_all_time", "revenue_net_all_time", "units_gross_all_time", "revenue_gross_all_time"]:
        if metric in SALES_FOR_PRODUCTS and metric in SALES_SUMMARY:
            matched = SALES_FOR_PRODUCTS["sales_row_present"]
            before = pd.to_numeric(SALES_SUMMARY.set_index("product_id").reindex(SELECTED.loc[matched, "product_id"])[metric], errors="coerce").sum(min_count=1)
            after = pd.to_numeric(SALES_FOR_PRODUCTS.loc[matched, metric], errors="coerce").sum(min_count=1)
            same = (pd.isna(before) and pd.isna(after)) or np.isclose(before, after, rtol=1e-8, atol=1e-6)
            validation(f"sales totals unchanged by product join ({metric})", same, after, before,
                       detail="Each selected product is joined once after aggregating source rows by product id.")

# Report uncategorized allowed attributes / rules that have no provider configured.
UNPROCESSED_ATTRIBUTES = []
for target in CFG["targets"]:
    rk = _category_rule_key(target)
    for attr, values in ALLOWED[MATRIX_CATEGORY[target]].items():
        if attr not in PIPELINE.get(rk, {}) and attr not in DEFAULT_PIPELINE and attr not in DIRECT_ATTRIBUTE_ALIASES:
            # generic exact-text fallback still runs; this is a reportable limitation, not a missing assignment row.
            UNPROCESSED_ATTRIBUTES.append({"subcategory": target, "attribute": attr, "allowed_values": " | ".join(values),
                                           "limitation": "No category-specific rule/provider. Generic exact-value text fallback only; review the rule config."})
        # identify a provider name that is not implemented/configured (new providers require code).
        for provider in PIPELINE.get(rk, {}).get(attr, []):
            implemented = {"style_fewshot", "style_zeroshot", "style_text_hint", "keywords", "vase_form", "table_seats", "lounge_seat_width",
                           "colour_text", "colour_image", "image_zeroshot", "column_map", "generic_text", "direct_attribute"}
            if provider not in implemented:
                UNPROCESSED_ATTRIBUTES.append({"subcategory": target, "attribute": attr, "allowed_values": " | ".join(values),
                                               "limitation": f"Configured provider {provider!r} is not implemented."})

VALIDATION_REPORT = pd.DataFrame(VALIDATION_ROWS)
FAILURES = _v_failures()
if len(FAILURES):
    dq("validation", "structural validation checks failed", len(FAILURES), "error", "; ".join(r["check"] for r in FAILURES))
else:
    dq("validation", "structural validation checks failed", 0, "info")
if UNPROCESSED_ATTRIBUTES:
    dq("validation", "attributes with only generic fallback or unimplemented providers", len(UNPROCESSED_ATTRIBUTES), "warning")

# Assignment quality warnings by attribute.
_not_det_share = ASSIGNMENTS.groupby(["subcategory", "attribute_name"])["is_sentinel"].mean()
for (cat, attr), share in _not_det_share.items():
    if share > CFG["validation"]["max_not_determinable_share_warn"]:
        dq("assignment", "high not-determinable share", int(round(100*share)), "warning", f"{cat}/{attr}: {share:.1%}")

display(VALIDATION_REPORT)
if UNPROCESSED_ATTRIBUTES: display(pd.DataFrame(UNPROCESSED_ATTRIBUTES))
if FAILURES and CFG["validation"]["strict"]:
    raise AssertionError("Structural validation failed: " + "; ".join(r["check"] for r in FAILURES))

# %% [markdown]
# ## 12. Export results and summary reports
# CSVs use UTF-8 with BOM for Excel compatibility. Each product-level join is validated one-to-one; sales stay missing where no source row exists.

# %%
# ---------- product-level pivot ----------------------------------------------------------------

def _is_internal_col(c):
    return str(c).startswith("_") or c in {"sub_category_key", "image_urls", "length_cm", "depth_cm", "height_cm", "weight_g"}

base = SELECTED[[c for c in SELECTED.columns if not _is_internal_col(c)]].copy()
base["primary_image_url"] = base["product_id"].map(lambda p: IMAGE_LINK_MAP.get(p, [None])[0] if IMAGE_LINK_MAP.get(p) else None)

# required assigned attributes, one column per attribute name
wide_attrs = ASSIGNMENTS.pivot(index="product_id", columns="attribute_name", values="assigned_value").reset_index()
wide_attrs.columns.name = None
PRODUCT_WIDE = base.merge(wide_attrs, on="product_id", how="left", validate="one_to_one")

# Sales summary joined to the product-level matrix. 'subcategory' from sales is redundant, so drop it first.
_sales_join = SALES_FOR_PRODUCTS.drop(columns=["subcategory"], errors="ignore")
PRODUCT_SALES = PRODUCT_WIDE.merge(_sales_join, on="product_id", how="left", validate="one_to_one", suffixes=("", "_sales"))
assert len(PRODUCT_WIDE) == len(SELECTED) == len(PRODUCT_SALES)
assert not PRODUCT_WIDE["product_id"].duplicated().any()
assert not PRODUCT_SALES["product_id"].duplicated().any()

# ---------- summaries --------------------------------------------------------------------------
PRODUCT_COUNTS_SUMMARY = SELECTED.groupby("subcategory", sort=False).agg(product_count=("product_id", "nunique")).reindex(CFG["targets"].keys(), fill_value=0).reset_index()
ATTRIBUTE_COMPLETION_SUMMARY = ASSIGNMENT_SUMMARY.copy()
CONFIDENCE_DISTRIBUTION = (ASSIGNMENTS.groupby("confidence", sort=False).size().rename("assignment_count").reset_index())
CONFIDENCE_DISTRIBUTION["percentage"] = (100 * CONFIDENCE_DISTRIBUTION["assignment_count"] / max(len(ASSIGNMENTS),1)).round(2)
UNCERTAIN_PRODUCTS = (UNCERTAIN_ASSIGNMENTS.groupby(["product_id", "product_name", "subcategory"], sort=False)
                      .agg(uncertain_attributes=("attribute_name", lambda x: " | ".join(x)),
                           low_or_missing_count=("attribute_name", "size"),
                           assignment_status=("confidence", lambda x: " | ".join(sorted(set(x)))))
                      .reset_index())

# Every allowed option is reported, including options with zero assignments. A zero count means no evidence-supported assignment,
# not that the option is absent from the catalogue.
_value_rows = []
for target in CFG["targets"]:
    for attribute, allowed in ALLOWED[MATRIX_CATEGORY[target]].items():
        sub = ASSIGNMENTS[(ASSIGNMENTS["subcategory"] == target) & (ASSIGNMENTS["attribute_name"] == attribute)]
        is_multi = attribute in CFG["assignment"]["multi_value_attributes"].get(_category_rule_key(target), [])
        observed = []
        for raw_val in sub.loc[~sub["is_sentinel"], "assigned_value"].dropna().astype(str):
            observed.extend(raw_val.split(CFG["assignment"]["multi_value_separator"]) if is_multi else [raw_val])
        counts = Counter(observed)
        for value in allowed:
            count = int(counts.get(value, 0))
            _value_rows.append({"subcategory": target, "attribute_name": attribute, "allowed_value": value,
                                "assigned_count": count, "processed_in_selection": count > 0,
                                "note": "" if count else "No product was assigned this value from available evidence; this does not prove it is absent."})
ALLOWED_VALUE_COVERAGE = pd.DataFrame(_value_rows)
UNASSIGNED_ALLOWED_VALUES = ALLOWED_VALUE_COVERAGE[~ALLOWED_VALUE_COVERAGE["processed_in_selection"]].copy()

# Sales top sellers. Empty period file is deliberate when the sales source cannot filter by date.
TOP_SELLERS_REQUESTED_PERIOD = pd.DataFrame(columns=["product_id", "product_name", "subcategory", "units_sold_requested_period",
                                                    "revenue_requested_period", "sales_rank_requested_period", "period_status", "note"])
if SALES_PERIOD_STATUS != "date_column_found_period_filtered":
    TOP_SELLERS_REQUESTED_PERIOD = pd.DataFrame([{"product_id": "", "product_name": "", "subcategory": cat,
        "units_sold_requested_period": np.nan, "revenue_requested_period": np.nan, "sales_rank_requested_period": np.nan,
        "period_status": SALES_PERIOD_STATUS, "note": "Unavailable: the source has no usable date column, so no product can be ranked for this period."}
        for cat in CFG["targets"]])
if SALES_PERIOD_STATUS == "date_column_found_period_filtered" and "units_sold_requested_period" in PRODUCT_SALES:
    sort_col = "units_sold_requested_period" if CFG["sales"]["rank_by"] == "units" else "revenue_requested_period"
    TOP_SELLERS_REQUESTED_PERIOD = (PRODUCT_SALES.sort_values(sort_col, ascending=False, na_position="last")
                                    .groupby("subcategory", sort=False).head(10)
                                    [["product_id", "product_name", "subcategory", "units_sold_requested_period", "revenue_requested_period", "sales_rank_requested_period"]]
                                    .copy())
    TOP_SELLERS_REQUESTED_PERIOD["period_status"] = SALES_PERIOD_STATUS
    TOP_SELLERS_REQUESTED_PERIOD["note"] = "Top products ranked only within the configured inclusive date period."

if "sales_rank_all_time_proxy" in PRODUCT_SALES:
    proxy_metric = "units_net_all_time" if "units_net_all_time" in PRODUCT_SALES else "units_gross_all_time"
    TOP_SELLERS_ALL_TIME_PROXY = (PRODUCT_SALES.sort_values(proxy_metric, ascending=False, na_position="last")
                                  .groupby("subcategory", sort=False).head(10)
                                  [["product_id", "product_name", "subcategory", proxy_metric, "revenue_net_all_time" if "revenue_net_all_time" in PRODUCT_SALES else proxy_metric,
                                    "sales_rank_all_time_proxy"]].copy())
    TOP_SELLERS_ALL_TIME_PROXY["note"] = "All-time aggregate proxy only; not the requested Nov 2025-Oct 2026 period."
else: TOP_SELLERS_ALL_TIME_PROXY = pd.DataFrame()

# Product-level join health and limitations.
JOIN_REPORT = pd.DataFrame([
    {"check": "selected products", "count": len(SELECTED), "detail": "Unique product IDs selected by configured sub-category mapping."},
    {"check": "products with any existing ItemAttribute rows", "count": int(SELECTED["existing_attribute_count"].gt(0).sum()), "detail": "At least one ItemAttribute row matched by product ID."},
    {"check": "products with at least one image link", "count": int(SELECTED["image_link_count"].gt(0).sum()), "detail": "At least one ItemMedia URL matched by product ID."},
    {"check": "products with usable downloaded/cached images", "count": int(SELECTED["product_id"].isin(set(IMAGE_FILES.loc[IMAGE_FILES["status"].isin(["cached", "downloaded"]), "product_id"])) .sum()) if len(IMAGE_FILES) else 0,
     "detail": "Valid decoded image exists in the local cache."},
    {"check": "products with no sales row", "count": int((~SALES_FOR_PRODUCTS.get("sales_row_present", pd.Series(False, index=SALES_FOR_PRODUCTS.index))).sum()),
     "detail": "Missing source row, not assumed zero sales."},
    {"check": "category conflicts", "count": int(CATEGORY_CHECK["category_conflict"].sum()), "detail": "DWH subcategory vs ItemAttribute Shopify Sub Category; reviewed, not auto-reclassified."},
])

# Final notebook summary tables (also exported as separate CSVs below).
display(PRODUCT_COUNTS_SUMMARY)
display(ATTRIBUTE_COMPLETION_SUMMARY)
display(CONFIDENCE_DISTRIBUTION)
display(UNCERTAIN_PRODUCTS.head(30))
display(TOP_SELLERS_REQUESTED_PERIOD)
if len(TOP_SELLERS_ALL_TIME_PROXY): display(TOP_SELLERS_ALL_TIME_PROXY)
display(UNASSIGNED_ALLOWED_VALUES)

# ---------- quality report / run metadata -------------------------------------------------------
DATA_QUALITY_REPORT = pd.DataFrame(DQ)
if len(VALIDATION_REPORT):
    _validation_dq = VALIDATION_REPORT.copy()
    _validation_dq["area"] = "validation"
    _validation_dq["check"] = "structural: " + _validation_dq["check"].astype(str)
    _validation_dq["count"] = _validation_dq["observed"]
    _validation_dq["detail"] = _validation_dq["detail"].fillna("") + " expected=" + _validation_dq["expected"].astype(str)
    _validation_dq = _validation_dq[["area", "check", "count", "severity", "detail"]]
    _dq = DATA_QUALITY_REPORT.copy()
    for col in ["area", "check", "count", "severity", "detail"]:
        if col not in _dq: _dq[col] = ""
    DATA_QUALITY_REPORT = pd.concat([_dq[["area", "check", "count", "severity", "detail"]], _validation_dq], ignore_index=True)

LIMITATIONS = []
if PDF_LIMITATION: LIMITATIONS.append(PDF_LIMITATION)
if not any(PDF_EXAMPLES.get("rule_key", pd.Series(dtype=str)).eq("vase")) if len(PDF_EXAMPLES) else True:
    LIMITATIONS.append("No Vase examples in Designstyles.pdf; Vase style confidence is capped at Low and uses generic prompts/weak transferred similarity only.")
if SALES_PERIOD_STATUS != "date_column_found_period_filtered":
    LIMITATIONS.append("Supplied sales data has no usable date column; requested-period metrics/ranks are missing (not zero). All-time aggregate columns are separate proxy context only.")
if not SALES_CURRENCY_VERIFIED: LIMITATIONS.append(f"Sales currency is not stated in source; configured {SALES_CURRENCY} is unverified.")
if not len(CROP_FILES): LIMITATIONS.append("No usable product images were downloaded/cached; visual evidence is unavailable for this run.")
if MODEL_BACKEND == "classic": LIMITATIONS.append("Pretrained SigLIP/DINOv2 checkpoints were unavailable; visual similarity uses classical HOG/HSV features (not semantic embeddings). Install optional model dependencies and allow model checkpoint access for deep embeddings.")
if not len(UNPROCESSED_ATTRIBUTES): pass
else: LIMITATIONS.append(f"{len(UNPROCESSED_ATTRIBUTES)} category-attribute combination(s) rely on generic fallback or need an unimplemented provider; see unprocessed_attributes.csv.")

run_metadata = {
    "project_dir": str(PROJECT_DIR), "source_dir": str(SRC_DIR), "result_dir": str(RES_DIR),
    "source_files": {k: str(find_file(k)) if CFG["files"].get(k) else None for k in ["products", "item_attributes", "media", "sales", "allowed_values", "pdf"]},
    "selected_products": int(len(SELECTED)), "selected_by_subcategory": dict(zip(PRODUCT_COUNTS_SUMMARY["subcategory"], PRODUCT_COUNTS_SUMMARY["product_count"].astype(int))),
    "assignment_rows": int(len(ASSIGNMENTS)), "assignments_by_confidence": {str(k): int(v) for k,v in ASSIGNMENTS["confidence"].value_counts().items()},
    "model_backend": MODEL_BACKEND, "model_name": ENCODER.name, "deep_embedding_available": MODEL_BACKEND in ("transformers", "open_clip", "dinov2"),
    "pdf_examples": int(len(PDF_EXAMPLES)), "downloaded_or_cached_images": int(IMAGE_FILES["status"].isin(["downloaded", "cached"]).sum()) if len(IMAGE_FILES) else 0,
    "failed_image_downloads": int(len(FAILED_IMAGES)), "sales_period": SALES_PERIOD_REPORT,
    "limitations": LIMITATIONS,
}

# ---------- write all requested outputs --------------------------------------------------------
def _write_csv(df, filename):
    path = RES_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

OUTPUTS = []
OUTPUTS.append(_write_csv(ASSIGNMENTS, "product_attribute_assignments.csv"))
OUTPUTS.append(_write_csv(PRODUCT_WIDE, "product_matrix_wide.csv"))
OUTPUTS.append(_write_csv(PRODUCT_SALES, "product_sales_summary.csv"))
OUTPUTS.append(_write_csv(DATA_QUALITY_REPORT, "data_quality_report.csv"))
OUTPUTS.append(_write_csv(VALIDATION_REPORT, "validation_report.csv"))
OUTPUTS.append(_write_csv(PRODUCT_COUNTS_SUMMARY, "summary_products_by_subcategory.csv"))
OUTPUTS.append(_write_csv(pd.DataFrame(CATEGORY_MAP_ROWS), "category_mapping_report.csv"))
OUTPUTS.append(_write_csv(ATTRIBUTE_COMPLETION_SUMMARY, "summary_assignment_completion_by_attribute.csv"))
OUTPUTS.append(_write_csv(CONFIDENCE_DISTRIBUTION, "summary_confidence_distribution.csv"))
OUTPUTS.append(_write_csv(UNCERTAIN_PRODUCTS, "products_with_uncertain_assignments.csv"))
OUTPUTS.append(_write_csv(JOIN_REPORT, "summary_join_health.csv"))
OUTPUTS.append(_write_csv(TOP_SELLERS_REQUESTED_PERIOD, "top_sellers_requested_period.csv"))
OUTPUTS.append(_write_csv(TOP_SELLERS_ALL_TIME_PROXY, "top_sellers_all_time_proxy.csv"))
OUTPUTS.append(_write_csv(pd.DataFrame(UNPROCESSED_ATTRIBUTES, columns=["subcategory", "attribute", "allowed_values", "limitation"]), "unprocessed_attributes.csv"))
OUTPUTS.append(_write_csv(ALLOWED_VALUE_COVERAGE, "allowed_value_coverage.csv"))
OUTPUTS.append(_write_csv(UNASSIGNED_ALLOWED_VALUES, "unassigned_allowed_values.csv"))
OUTPUTS.append(_write_csv(SELECTED_ATTR, "selected_existing_attributes_long.csv"))
OUTPUTS.append(_write_csv(IMAGE_FILES, "image_download_status.csv"))
OUTPUTS.append(_write_csv(IMAGE_RECORDS_DF, "product_image_crop_manifest.csv"))
OUTPUTS.append(_write_csv(PDF_EXAMPLES, "pdf_example_manifest.csv"))
OUTPUTS.append(_write_csv(PDF_RULES, "pdf_page_style_rules.csv"))

# Parquet is optional, convenient for larger results.
try:
    ASSIGNMENTS.to_parquet(RES_DIR / "product_attribute_assignments.parquet", index=False)
    PRODUCT_SALES.to_parquet(RES_DIR / "product_sales_summary.parquet", index=False)
    OUTPUTS.extend([RES_DIR / "product_attribute_assignments.parquet", RES_DIR / "product_sales_summary.parquet"])
except Exception as e:
    log(f"Parquet export unavailable; CSVs were written ({type(e).__name__}: {e})", "WARN")

(RES_DIR / "sales_period_method.json").write_text(json.dumps(_json_safe(SALES_PERIOD_REPORT), indent=2, ensure_ascii=False), encoding="utf-8")
(RES_DIR / "run_metadata.json").write_text(json.dumps(_json_safe(run_metadata), indent=2, ensure_ascii=False), encoding="utf-8")

# Processing log, including explicit row counts, skips, failures, joins, and limitations.
log("================ PROCESSING SUMMARY ================")
log(f"source product rows={len(SOURCES['products']):,}; unique product IDs after cleanup={len(PRODUCTS_ALL):,}; selected target products={len(SELECTED):,}")
log(f"assignments={len(ASSIGNMENTS):,}; missing/uncertain product-attribute rows={int(ASSIGNMENTS['is_sentinel'].sum()):,} not-determinable; low-confidence rows={int((ASSIGNMENTS['confidence']=='Low').sum()):,}")
log(f"duplicate products removed={next((x['count'] for x in DQ if x['check']=='rows sharing a product id'), 0)}; duplicate ItemAttribute rows removed={next((x['count'] for x in DQ if x['check']=='exact duplicate rows (removed)'), 0)}")
log(f"products without ItemAttribute rows={len(missing_attr):,}; products without image links={len(missing_img):,}; failed image links={len(FAILED_IMAGES):,}; usable images={len(CROP_FILES):,}")
log(f"sales period status={SALES_PERIOD_STATUS}; selected-product sales joins={int(SALES_FOR_PRODUCTS.get('sales_row_present', pd.Series(False,index=SALES_FOR_PRODUCTS.index)).sum()):,}; unmatched/missing sales rows={int((~SALES_FOR_PRODUCTS.get('sales_row_present', pd.Series(False,index=SALES_FOR_PRODUCTS.index))).sum()):,}")
for lim in LIMITATIONS: log("LIMITATION: " + lim, "WARN")
for out in OUTPUTS: log("OUTPUT: " + out.name)
(RES_DIR / "processing_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")

# Data-quality report contains findings generated by the export section above as well.
DATA_QUALITY_REPORT = pd.DataFrame(DQ)
_write_csv(DATA_QUALITY_REPORT, "data_quality_report.csv")

print("\nCreated result files:")
for out in OUTPUTS: print(" -", out)
print("\nRequested period:", SALES_PERIOD_STATUS, f"({SALES_PERIOD_START.date()} through {SALES_PERIOD_END.date()})")
print("Structural validation failures:", len(FAILURES))

# %% [markdown]
# ## README – how to run and extend the Category Matrix notebook
#
# ### Project layout
# Put the notebook in `Category Matrix/` beside `SRC/`, with the six source files inside `SRC/`. It creates `RES/` automatically.
# If the notebook is elsewhere, set `PROJECT_DIR_OVERRIDE` in Section 1 to the full project-folder path.
#
# ### Run
# 1. Open the notebook in Jupyter.
# 2. Edit **Section 1a** for paths, target category names, source column mappings, date period, image limits, model IDs and device.
# 3. Edit **Section 1b** only if you need new attribute rules, image descriptions, text patterns, or price/tier mappings.
# 4. Choose **Kernel → Restart & Run All**. Re-running uses cached valid images, PDF crops and embeddings.
#
# The required data packages are `pandas`, `numpy`, `Pillow`, `requests`, `PyMuPDF` (`fitz`), and `scikit-learn`; `opencv-python`
# improves foreground cropping and colour analysis; `pyarrow` enables Parquet. These common packages are optional where a fallback exists.
# For pretrained photo embeddings and object-aware crops, install compatible `torch` + `transformers` (and optionally `open_clip_torch`
# or `rembg` + `onnxruntime`). The first deep-model run needs the configured pretrained checkpoint files locally or access to download them;
# for offline model use set `CFG['models']['local_files_only']=True` and pre-cache the checkpoints. The notebook never installs packages or
# embeds credentials automatically. A server that blocks image downloads yields a clear status report and a text-only / existing-cache run.
#
# ### Adding future attributes (e.g. price, tier)
# - Add the attribute and its valid values to `category_attribute_values_vertical.csv` for the appropriate category.
# - Add the new product column to `dwh_present_select.csv` (or configure a one-row-per-product file under `CFG['extra_sources']`).
# - The notebook automatically carries new unmapped product columns. For a direct exact mapping, value map, or numeric bins, add one
#   entry to `ATTRIBUTE_FROM_COLUMN` in Section 1b. Example: map the detected logical `tier` column to the allowed `Price tier` attribute.
# - For evidence that is semantic rather than exact, add a category-specific provider/rule to `PIPELINE` / `KEYWORD_RULES` / `IMAGE_PROMPTS`.
#   An attribute with no dedicated rule still receives a conservative generic exact-text fallback and is listed in `unprocessed_attributes.csv`.
# - Column names are matched ignoring capitalization, spaces and separators; edit `CFG['columns']` or the source candidate list if yours differs.
#
# ### Image similarity
# The preferred path uses SigLIP joint image/text embeddings to compare cropped photos against attribute descriptions and DINOv2 image
# embeddings to compare product appearance against the extracted, specialist-labelled PDF examples. Optional OWLv2 detects/crops the target
# product in lifestyle images. `rembg` is an optional background-removal fallback. If unavailable, the notebook records the fallback and uses
# classical HOG/HSV descriptors (colour/shape only), not semantic embeddings. `image_download_status.csv` and `product_image_crop_manifest.csv`
# retain image failures and crop provenance; embedding cache files are under `RES/cache_embeddings/`.
#
# ### Confidence interpretation
# For each candidate allowed value, the notebook keeps the strongest support inside each evidence family so repeated source fields do not
# inflate confidence. Independent evidence-family scores combine using noisy-OR. The strongest competing value incurs the configured conflict
# penalty. Thresholds are High (0.75+), Medium (0.50–0.749), Low (0.30–0.499), and below 0.30 is `Not determinable`. Subjective design style
# is capped at Medium unless the product is near-identical to a labelled PDF example, and at Low for categories without their own examples.
# Confidence is a reproducible evidence score, not a calibrated probability of correctness; low/uncertain assignments are exported for review.
#
# ### Sales and known limitation in the supplied files
# Sales aggregation uses the mapped product key and measures documented in `RES/sales_period_method.json`. Requested period: **Nov 2025–Oct 2026 inclusive**.
# The attached sales input has aggregate product quantities/revenues but no sale date or order-ID column, so period-filtered units, revenue,
# order count, first/last date, and period rank cannot be calculated. Those fields remain missing—not zero. All-time source aggregates and their
# per-subcategory ranks are provided as an explicitly marked proxy only. Source currency is not stated; the configured currency is unverified.
#
# ### Output files in `RES/`
# - `product_attribute_assignments.csv` (and optional `.parquet`): long format, one product × required attribute, valid value/sentinel,
#   confidence, evidence and source.
# - `product_matrix_wide.csv`: one row per product, all assigned attributes plus available source fields.
# - `product_sales_summary.csv` (and optional `.parquet`): wide product matrix joined once to sales context.
# - `data_quality_report.csv`, `validation_report.csv`, `processing_log.txt`, `run_metadata.json`, `sales_period_method.json`.
# - Summary tables: category mapping, product counts, completion by attribute, confidence distribution, uncertain products, join health, allowed-value coverage/unassigned values, requested-period top sellers
#   (explicit unavailable status when dates are absent), and all-time top-selling proxy.
# - `image_download_status.csv`, PDF example manifests/rules, selected existing attributes; caches and intermediate crops are in named subfolders.
#
# The workflow does not invent missing product facts. Review low-confidence and `Not determinable` rows, correct the source evidence or rules,
# and re-run for a fully auditable refresh.
