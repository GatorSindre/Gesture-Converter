"""
Gesture Typing Database Builder with Automatic Word List Download
Downloads comprehensive word lists and frequency data from online sources.
"""

import sqlite3
import os
import requests
from typing import Dict, Optional, List
from collections import Counter

# ============ CONFIGURATION ============
DB_PATH = "patterns.db"
WORD_SOURCES = {
    # Comprehensive English word list (~370k words)
    'dwyl_words': 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt',
    
    # Common words with frequency (top 10k)
    'google_10k': 'https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt',
    
    # Additional common words
    'common_words': 'https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt',
}

# ============ FINGER MAPPING ============
LETTER_TO_FINGER = {
    'space': '0', 'c': '0',           # Left thumb
    'v': '1', 'b': '1', 'r': '1', 'f': '1', 't': '1', 'g': '1',  # Left index
    'e': '2', 'd': '2',               # Left middle
    'w': '3', 's': '3', 'x': '3',     # Left ring
    'q': '4', 'a': '4', 'z': '4',     # Left pinky
    'n': '5', 'm': '5',               # Right thumb
    'y': '6', 'h': '6', 'j': '6', 'u': '6',  # Right index
    'i': '7', 'k': '7', ',': '7',     # Right middle
    'o': '8', 'l': '8', '.': '8',     # Right ring
    'p': '9',                          # Right pinky
}

# ============ WORD LIST DOWNLOADING ============
def download_word_list(url: str, name: str) -> List[str]:
    """Download a word list from a URL."""
    try:
        print(f"📥 Downloading {name}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        words = [line.strip().lower() for line in response.text.split('\n') if line.strip()]
        print(f"✅ Downloaded {len(words)} words from {name}")
        return words
    
    except requests.RequestException as e:
        print(f"⚠️  Failed to download {name}: {e}")
        return []

def download_all_word_lists() -> Dict[str, List[str]]:
    """Download word lists from all configured sources."""
    print("=" * 60)
    print("🌐 DOWNLOADING WORD LISTS FROM ONLINE SOURCES")
    print("=" * 60)
    
    all_lists = {}
    for name, url in WORD_SOURCES.items():
        words = download_word_list(url, name)
        if words:
            all_lists[name] = words
    
    return all_lists

def merge_word_lists(word_lists: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Merge multiple word lists and assign frequency scores.
    Words appearing in multiple sources get higher frequency.
    Earlier lists (like google_10k) get priority.
    """
    word_frequency = Counter()
    
    # Process google_10k first (these are ranked by frequency)
    if 'google_10k' in word_lists:
        for idx, word in enumerate(word_lists['google_10k'][:10000]):
            # Higher rank = higher frequency (inverse of index)
            word_frequency[word] = 10000 - idx
    
    # Process common words list
    if 'common_words' in word_lists:
        for word in word_lists['common_words']:
            if word not in word_frequency:
                word_frequency[word] = 5000
            else:
                word_frequency[word] += 2000
    
    # Process comprehensive list (lower priority)
    if 'dwyl_words' in word_lists:
        for word in word_lists['dwyl_words']:
            if word not in word_frequency:
                word_frequency[word] = 100
            else:
                word_frequency[word] += 50
    
    # Ensure minimum length and valid characters
    filtered = {
        word: freq 
        for word, freq in word_frequency.items() 
        if len(word) >= 2 and word.isalpha()
    }
    
    print(f"\n📊 Merged {len(filtered)} unique valid words")
    return filtered

# ============ PATTERN GENERATION ============
def word_to_pattern(word: str) -> Optional[str]:
    """
    Convert a word to its finger pattern.
    
    Example: "hello" -> "6-2-8-8-8"
    """
    word = word.lower().strip()
    fingers = []
    
    for char in word:
        finger = LETTER_TO_FINGER.get(char)
        if finger is None:
            return None
        fingers.append(finger)
    
    return "-".join(fingers)

# ============ DATABASE OPERATIONS ============
class DatabaseBuilder:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def connect(self) -> None:
        """Connect to the database."""
        if os.path.exists(self.db_path):
            print(f"🗑️  Removing existing database: {self.db_path}")
            os.remove(self.db_path)
        
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        print(f"✅ Connected to database: {self.db_path}")
    
    def create_schema(self) -> None:
        """Create the database schema."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL UNIQUE,
                pattern TEXT NOT NULL,
                frequency INTEGER DEFAULT 1
            )
        """)
        
        # Create indexes for fast lookups
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern 
            ON words(pattern, frequency DESC)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_word
            ON words(word)
        """)
        
        self.conn.commit()
        print("✅ Database schema created")
    
    def bulk_insert(self, word_data: List[tuple]) -> None:
        """Bulk insert multiple words."""
        # Use INSERT OR IGNORE to skip duplicates
        self.cursor.executemany("""
            INSERT OR IGNORE INTO words (word, pattern, frequency)
            VALUES (?, ?, ?)
        """, word_data)
        self.conn.commit()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        self.cursor.execute("SELECT COUNT(*) FROM words")
        total_words = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(DISTINCT pattern) FROM words")
        unique_patterns = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT AVG(frequency) FROM words")
        avg_frequency = self.cursor.fetchone()[0]
        
        return {
            'total_words': total_words,
            'unique_patterns': unique_patterns,
            'avg_frequency': int(avg_frequency) if avg_frequency else 0
        }
    
    def get_top_words(self, limit: int = 20) -> List[tuple]:
        """Get top words by frequency."""
        self.cursor.execute("""
            SELECT word, pattern, frequency
            FROM words
            ORDER BY frequency DESC
            LIMIT ?
        """, (limit,))
        return self.cursor.fetchall()
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")

# ============ MAIN BUILD PROCESS ============
def build_database_from_online_sources() -> None:
    """Build the gesture typing database from online word lists."""
    print("=" * 60)
    print("🏗️  GESTURE TYPING DATABASE BUILDER")
    print("🌐 AUTOMATIC WORD LIST DOWNLOAD")
    print("=" * 60)
    
    # Download word lists
    word_lists = download_all_word_lists()
    
    if not word_lists:
        print("❌ Failed to download any word lists. Check your internet connection.")
        return
    
    # Merge and score words
    print(f"\n🔄 Merging word lists...")
    word_frequency = merge_word_lists(word_lists)
    
    # Initialize database
    db = DatabaseBuilder()
    db.connect()
    db.create_schema()
    
    # Process words in batches
    print(f"\n🔄 Processing {len(word_frequency)} words...")
    word_data = []
    skipped = 0
    batch_size = 10000
    
    for word, frequency in word_frequency.items():
        pattern = word_to_pattern(word)
        if pattern:
            word_data.append((word, pattern, frequency))
            
            # Insert in batches for better performance
            if len(word_data) >= batch_size:
                print(f"💾 Inserting batch of {len(word_data)} words...")
                db.bulk_insert(word_data)
                word_data = []
        else:
            skipped += 1
    
    # Insert remaining words
    if word_data:
        print(f"💾 Inserting final batch of {len(word_data)} words...")
        db.bulk_insert(word_data)
    
    # Statistics
    stats = db.get_statistics()
    print("\n" + "=" * 60)
    print("📊 DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total words inserted: {stats['total_words']:,}")
    print(f"Unique patterns: {stats['unique_patterns']:,}")
    print(f"Average frequency: {stats['avg_frequency']:,}")
    print(f"Words skipped (unmapped chars): {skipped:,}")
    print(f"Database file: {db.db_path}")
    print(f"Database size: {os.path.getsize(db.db_path) / 1024 / 1024:.2f} MB")
    
    # Show top words
    print("\n📋 Top 20 words by frequency:")
    top_words = db.get_top_words(20)
    for word, pattern, freq in top_words:
        print(f"  {word:15} → {pattern:25} (freq: {freq:,})")
    
    db.close()
    print("\n✅ Database build complete!")
    print("💡 You can now run: python main.py")

# ============ ENTRY POINT ============
if __name__ == "__main__":
    import sys
    
    # Check if requests is installed
    try:
        import requests
    except ImportError:
        print("❌ Error: 'requests' library not found")
        print("📦 Install it with: pip install requests")
        sys.exit(1)
    
    build_database_from_online_sources()