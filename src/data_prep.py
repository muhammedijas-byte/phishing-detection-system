import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import RAW_DATA_PATH, TARGET_COL, RANDOM_STATE
from urllib.parse import urlparse
import tldextract



def load_data(path=None):
    """Load phishing dataset."""
    csv_path = RAW_DATA_PATH if path is None else path
    df = pd.read_csv(csv_path)
    print("[INFO] Loaded dataset from:", csv_path)
    print("[INFO] Raw dataset shape:", df.shape)

    # 🔹 NEW: convert all numeric-looking columns to real numbers
    df = df.apply(pd.to_numeric, errors="ignore")

    print("[INFO] Dataset dtypes summary:")
    print(df.dtypes.value_counts())
    return df

def get_registered_domain(url: str) -> str:
    """
    Extract registered domain (example: mail.google.com → google.com)
    """
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    ext = tldextract.extract(parsed.netloc)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""



def prepare_data(df):
    # Separate features and label
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])
    print("[INFO] Using numeric feature columns:", list(X.columns))

    # Handle missing values
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
# Paste this into src/data_prep.py (append or merge with your file)
import re
import time
import math
import socket
import ipaddress
from urllib.parse import urlparse, unquote
import requests
from bs4 import BeautifulSoup
import tldextract
import pandas as pd

# User-agent and timeouts for polite scraping
_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PhishDetector/1.0; +https://example.local)"}
_REQUEST_TIMEOUT = 8  # seconds


def _safe_request(url):
    """Perform GET with timeout and return response or None."""
    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT, allow_redirects=True)
        return resp
    except Exception:
        return None


def _is_ip(hostname):
    """Return True if hostname is an IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except Exception:
        return False


def extract_features_from_url(url: str) -> pd.DataFrame:
    # Normalize input
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url  # assume http if scheme missing
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = unquote(url)
    # tldextract to get domain/host parts
    tx = tldextract.extract(hostname)
    subdomain = tx.subdomain or ""
    domain = tx.domain or ""
    suffix = tx.suffix or ""
    host_without_port = hostname.split(":")[0] if ":" in hostname else hostname
    # Basic counts
    nb_dots = hostname.count(".")
    nb_www = 1 if "www" in subdomain.split(".") or hostname.startswith("www.") else 0
    nb_hyphens = hostname.count("-")
    nb_space = full_url.count(" ")
    nb_percent = full_url.count("%")
    nb_redirection = 0  # follow-up from requests history
    ip_flag = 1 if _is_ip(host_without_port) else 0

    # https_token: check whether string 'https' appears in domain/host (suspicious token)
    https_token = 1 if "https" in hostname.lower() else 0

    # Count of hyperlinks and safe_anchor detection from HTML
    nb_hyperlinks = 0
    safe_anchor = 0
    empty_title = 0
    domain_in_title = 0
    domain_with_copyright = 0
    phish_hints = 0
    avg_word_host = 0.0
    ratio_digits_host = 0.0

    # Attempt to fetch page
    resp = _safe_request(url)
    if resp is not None:
        # redirection count = history length (number of redirects)
        try:
            nb_redirection = len(resp.history) if hasattr(resp, "history") else 0
        except Exception:
            nb_redirection = 0

        # parse HTML if present
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type.lower() or True:
            try:
                soup = BeautifulSoup(resp.content, "html.parser")
                # hyperlinks: count of <a href=...>
                anchors = soup.find_all("a")
                nb_hyperlinks = len(anchors)

                # safe_anchor heuristic: ratio of anchors whose href points to same domain
                same_count = 0
                for a in anchors:
                    href = a.get("href")
                    if not href:
                        continue
                    href_parsed = urlparse(href)
                    # relative links considered same site
                    if href_parsed.netloc == "" or tx.domain in (tldextract.extract(href).domain or ""):
                        same_count += 1
                if nb_hyperlinks > 0:
                    safe_anchor = same_count / nb_hyperlinks
                else:
                    safe_anchor = 0.0

                # title presence & domain-in-title
                title_tag = soup.title.string if soup.title and soup.title.string else ""
                if title_tag is None or str(title_tag).strip() == "":
                    empty_title = 1
                else:
                    empty_title = 0
                    if domain.lower() in title_tag.lower():
                        domain_in_title = 1
                    else:
                        domain_in_title = 0

                # domain_with_copyright
                page_text = soup.get_text(separator=" ").lower()
                if "copyright" in page_text or "©" in page_text:
                    domain_with_copyright = 1
                else:
                    domain_with_copyright = 0

                # phish_hints: presence of suspicious words in url/path/title/text
                suspicious_words = ["login", "signin", "bank", "secure", "update", "verify", "confirm", "account", "webscr", "ebayisapi", "paypal"]
                found = 0
                hay = (full_url + " " + path + " " + query + " " + title_tag + " " + page_text).lower()
                for w in suspicious_words:
                    if w in hay:
                        found += 1
                phish_hints = found

            except Exception:
                # parsing failed - keep defaults
                pass
    else:
        # If request failed, keep many values as defaults / heuristics
        nb_hyperlinks = 0
        safe_anchor = 0.0
        empty_title = 1
        domain_in_title = 0
        domain_with_copyright = 0
        phish_hints = 0
        nb_redirection = 0

    # avg_word_host: average length of host parts (split on non alpha numeric)
    host_parts = re.split(r"[\W_]+", host_without_port)
    host_parts = [p for p in host_parts if p]
    if host_parts:
        avg_word_host = sum(len(p) for p in host_parts) / len(host_parts)
        digits = sum(ch.isdigit() for ch in host_without_port)
        ratio_digits_host = digits / max(1, len(host_without_port))
    else:
        avg_word_host = 0.0
        ratio_digits_host = 0.0

    # ip (numeric) - already computed as ip_flag
    ip = ip_flag

    # domain_age: cannot compute reliably without WHOIS; set 0 as fallback
    domain_age = 0.0
    # If WHOIS is available locally you could fill domain_age using python-whois; left as 0 for now.

    # page_rank, google_index, web_traffic - not available without third-party API; set 0
    page_rank = 0.0
    google_index = 0.0
    web_traffic = 0.0

    # nb_space, nb_percent done above
    # nb_www already 0/1; keep it numeric
    # nb_hyphens, nb_dots numeric

    # nb_percent, nb_space numeric
    # nb_redirection numeric

    # domain_in_title boolean -> convert to 0/1 already set

    # safe_anchor is ratio -> keep numeric float

    # Build a dictionary with the **selected features** in the same order used by your model
    # IMPORTANT: use your exact selected_features order when creating final DataFrame.
    features = {
        "page_rank": float(page_rank),
        "google_index": float(google_index),
        "nb_www": float(nb_www),
        "web_traffic": float(web_traffic),
        "phish_hints": float(phish_hints),
        "domain_age": float(domain_age),
        "nb_hyphens": float(nb_hyphens),
        "safe_anchor": float(safe_anchor),
        "nb_hyperlinks": float(nb_hyperlinks),
        "nb_space": float(nb_space),
        "nb_percent": float(nb_percent),
        "empty_title": float(empty_title),
        "domain_with_copyright": float(domain_with_copyright),
        "nb_dots": float(nb_dots),
        "domain_in_title": float(domain_in_title),
        "ip": float(ip),
        "avg_word_host": float(avg_word_host),
        "ratio_digits_host": float(ratio_digits_host),
        "nb_redirection": float(nb_redirection),
        "https_token": float(https_token),
    }

    # Return a single-row DataFrame
    df_out = pd.DataFrame([features])
    return df_out
