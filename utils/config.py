import os
import glob
import time
import pandas as pd
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class KeyManager:
    """
    Manages API key pool loaded dynamically from environment variables (.env).
    """
    def __init__(self):
        env_keys = os.getenv("GEMINI_API_KEYS", "").strip()
        if env_keys:
            self.keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        else:
            single = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            self.keys = [single] if single else []
            
        self.current_index = 0
        self.request_counts = {i: 0 for i in range(max(1, len(self.keys)))}
        self.last_reset = time.time()
        
    def add_key(self, new_key: str):
        """Adds a new API key to the pool if not already present."""
        clean_key = new_key.strip()
        if clean_key and clean_key not in self.keys:
            self.keys.append(clean_key)
            self.request_counts[len(self.keys) - 1] = 0

    def get_active_key(self) -> str:
        """Returns the currently active API key and sets system environment variables."""
        if not self.keys:
            return os.getenv("GEMINI_API_KEY", "")
        active_key = self.keys[self.current_index]
        os.environ["GEMINI_API_KEY"] = active_key
        os.environ["GOOGLE_API_KEY"] = active_key
        return active_key

    def rotate_to_next_key(self) -> Tuple[str, str]:
        """Rotates to the next available API key in pool when current key hits rate limit."""
        if len(self.keys) <= 1:
            return self.get_active_key(), "[Key Alert] Only 1 API key configured. Cannot switch to next key."
            
        old_idx = self.current_index
        self.current_index = (self.current_index + 1) % len(self.keys)
        new_key = self.get_active_key()
        msg = f"[Key Rotation] API Key #{old_idx + 1} quota reached. Automatically rotated to API Key #{self.current_index + 1}."
        return new_key, msg

    def record_request_and_check_warning(self) -> str:
        """Tracks request count per minute and returns warning if near quota (~80% capacity)."""
        now = time.time()
        if now - self.last_reset > 60:
            self.request_counts = {i: 0 for i in range(max(1, len(self.keys)))}
            self.last_reset = now
            
        if self.current_index not in self.request_counts:
            self.request_counts[self.current_index] = 0
            
        self.request_counts[self.current_index] += 1
        count = self.request_counts[self.current_index]
        
        if count >= 12 and len(self.keys) > 1:
            return f"[Quota Alert] API Key #{self.current_index + 1} is at {count}/15 req/min (~80%+ capacity). Failover key ready."
        return ""

# Initialize Global Key Manager
key_manager = KeyManager()

def get_gemini_api_key() -> str:
    """Retrieves current active API Key and ensures environment variables are set."""
    return key_manager.get_active_key()

# Set initial keys
get_gemini_api_key()

def get_dataset_path() -> str:
    """Locates the cleaned enterprise CSV dataset in the workspace."""
    candidates = [
        os.path.join(BASE_DIR, "data", "cleaned_enterprise_data_final.csv"),
        os.path.join(BASE_DIR, "data", "cleaned_enterprise_data.csv"),
        os.path.join(BASE_DIR, "cleaned_enterprise_data_final.csv"),
        os.path.join(BASE_DIR, "cleaned_enterprise_data.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
            
    csvs = glob.glob(os.path.join(BASE_DIR, "data", "*.csv"))
    if csvs:
        return csvs[0]
        
    raise FileNotFoundError("Could not locate enterprise dataset CSV file in workspace.")

def load_dataset() -> pd.DataFrame:
    """Loads the main dataframe into memory."""
    path = get_dataset_path()
    return pd.read_csv(path)

def get_gemini_client(api_key: str = None):
    """Initializes and returns a google.genai.Client instance."""
    from google import genai
    key = api_key or get_gemini_api_key()
    return genai.Client(api_key=key)
