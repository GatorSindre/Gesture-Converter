"""
Gesture Typing System with KenLM Language Model Integration
Uses context-aware ranking to suggest the most probable words based on sentence history.
"""

import keyboard
import time
import threading
import sqlite3
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass

# ============ CONFIGURATION ============
INACTIVITY_TIMEOUT = 1.0  # seconds before finalizing a word
DB_PATH = "patterns.db"
LM_PATH = "lm_model.arpa"  # KenLM language model file

# ============ FINGER MAPPING ============
# Original letter-to-finger mapping (used by database)
KEY_TO_FINGER = {
    "space": "0", "c": "0",           # Left thumb
    "v": "1", "b": "1", "r": "1", "f": "1", "t": "1", "g": "1",  # Left index
    "e": "2", "d": "2",               # Left middle
    "w": "3", "s": "3", "x": "3",     # Left ring
    "q": "4", "a": "4", "z": "4",     # Left pinky
    "n": "5", "m": "5",               # Right thumb
    "y": "6", "h": "6", "j": "6", "u": "6",  # Right index
    "i": "7", "k": "7", ",": "7",     #x Right middle
    "o": "8", "l": "8", ".": "8",     # Right ring
    "p": "9",                          # Right pinky
}

# Number key testing input (physical keyboard numbers -> finger codes)
NUMBER_TO_FINGER = {
    "1": "4",  # Left pinky
    "2": "3",  # Left ring
    "3": "2",  # Left middle
    "4": "1",  # Left index
    "5": "0",  # Left thumb
    "6": "5",  # Right thumb
    "7": "6",  # Right index
    "8": "7",  # Right middle
    "9": "8",  # Right ring
    "0": "9",  # Right pinky
}

# ============ LANGUAGE MODEL ============
class LanguageModel:
    """Wrapper for KenLM language model with fallback."""
    
    def __init__(self, model_path: str):
        self.model = None
        self.model_path = model_path
        self.use_kenlm = False
        
        # Try to load KenLM
        try:
            import kenlm
            if os.path.exists(model_path):
                self.model = kenlm.Model(model_path)
                self.use_kenlm = True
                print(f"✅ KenLM model loaded: {model_path}")
            else:
                print(f"⚠️  KenLM model not found: {model_path}")
                print("💡 Using frequency-based ranking (install model for better results)")
        except ImportError:
            print("⚠️  KenLM not installed")
            print("💡 Using frequency-based ranking")
            print("📦 To enable context-aware ranking, install KenLM:")
            print("   pip install https://github.com/kpu/kenlm/archive/master.zip")
    
    def score(self, sentence: str) -> float:
        """Score a sentence using the language model."""
        if self.use_kenlm and self.model:
            return self.model.score(sentence, bos=True, eos=True)
        else:
            # Fallback: simple length penalty (prefer shorter sentences)
            return -len(sentence.split())
    
    def score_next_word(self, context: List[str], candidate: str) -> float:
        """
        Score how likely a candidate word is to follow the given context.
        Higher scores = more probable.
        """
        if not context:
            # No context, just score the word alone
            return self.score(candidate)
        
        # Build sentence with candidate
        sentence = " ".join(context + [candidate])
        
        if self.use_kenlm and self.model:
            # Get full sentence score
            full_score = self.model.score(sentence, bos=True, eos=False)
            
            # Get context-only score
            context_sentence = " ".join(context)
            context_score = self.model.score(context_sentence, bos=True, eos=False)
            
            # Return the conditional probability (how much better with this word)
            return full_score - context_score
        else:
            # Fallback: prefer shorter words
            return -len(candidate)

# ============ STATE MANAGEMENT ============
@dataclass
class WordSuggestion:
    """Represents a word suggestion with scores."""
    word: str
    pattern: str
    frequency: int
    lm_score: float
    combined_score: float

class GestureState:
    def __init__(self):
        self.buffer: List[str] = []
        self.suggestions: List[WordSuggestion] = []
        self.final_sentence: List[str] = []
        self.last_key_time: float = time.monotonic()
        self.lock = threading.Lock()
    
    def add_finger(self, finger_code: str) -> None:
        """Add a finger code to the current word buffer."""
        with self.lock:
            self.buffer.append(finger_code)
            self.last_key_time = time.monotonic()
    
    def get_pattern(self) -> Optional[str]:
        """Get the current pattern as a hyphen-separated string."""
        with self.lock:
            if not self.buffer:
                return None
            return "-".join(self.buffer)
    
    def clear_buffer(self) -> None:
        """Clear the current word buffer."""
        with self.lock:
            self.buffer = []
    
    def is_inactive(self, timeout: float) -> bool:
        """Check if the buffer has been inactive for the timeout period."""
        with self.lock:
            return bool(self.buffer) and (time.monotonic() - self.last_key_time > timeout)

# ============ DATABASE INTERFACE ============
class PatternDatabase:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._verify_schema()
    
    def _verify_schema(self) -> None:
        """Verify that the database has the expected schema."""
        try:
            self.cursor.execute("SELECT word, pattern, frequency FROM words LIMIT 1")
        except sqlite3.OperationalError as e:
            print(f"⚠️  Database schema error: {e}")
            print("💡 Run: python database_builder.py")
    
    def lookup(self, pattern: str, limit: int = 20) -> List[Tuple[str, int]]:
        """Look up words matching the given pattern."""
        self.cursor.execute("""
            SELECT word, frequency
            FROM words
            WHERE pattern = ?
            ORDER BY frequency DESC
            LIMIT ?
        """, (pattern, limit))
        return self.cursor.fetchall()
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

# ============ MAIN LOGIC ============
class GestureTypingSystem:
    def __init__(self):
        self.state = GestureState()
        self.db = PatternDatabase(DB_PATH)
        self.lm = LanguageModel(LM_PATH)
        self.running = True
    
    def rank_candidates(self, candidates: List[Tuple[str, int]], context: List[str]) -> List[WordSuggestion]:
        """
        Rank candidate words using both frequency and language model scores.
        
        Combines:
        - Frequency score (from database)
        - Language model score (context probability)
        """
        suggestions = []
        
        for word, frequency in candidates:
            # Get language model score for this word in context
            lm_score = self.lm.score_next_word(context, word)
            
            # Normalize frequency (log scale to prevent overwhelming LM score)
            import math
            freq_score = math.log(frequency + 1) if frequency > 0 else 0
            
            # Combined score: weighted combination
            # Adjust weights based on whether we have KenLM
            if self.lm.use_kenlm:
                # With KenLM: prioritize language model (70%), frequency (30%)
                combined = (0.7 * lm_score) + (0.3 * freq_score)
            else:
                # Without KenLM: just use frequency
                combined = freq_score
            
            suggestions.append(WordSuggestion(
                word=word,
                pattern="",
                frequency=frequency,
                lm_score=lm_score,
                combined_score=combined
            ))
        
        # Sort by combined score (descending)
        suggestions.sort(key=lambda x: x.combined_score, reverse=True)
        
        return suggestions
    
    def lookup_suggestions(self, pattern: str) -> List[WordSuggestion]:
        """Look up and rank word suggestions for a pattern."""
        # Get candidates from database
        candidates = self.db.lookup(pattern, limit=20)
        
        if not candidates:
            print(f"\n❌ No matches for pattern: {pattern}")
            return []
        
        # Rank using language model
        context = self.state.final_sentence
        ranked_suggestions = self.rank_candidates(candidates, context)
        
        # Display suggestions
        print(f"\n💡 Top suggestions for pattern {pattern}:")
        for i, sugg in enumerate(ranked_suggestions[:10], 1):
            if self.lm.use_kenlm:
                print(f"  {i}. {sugg.word:15} (freq: {sugg.frequency:6}, LM: {sugg.lm_score:7.2f}, score: {sugg.combined_score:7.2f})")
            else:
                print(f"  {i}. {sugg.word:15} (freq: {sugg.frequency:6})")
        
        return ranked_suggestions
    
    def finalize_word(self) -> None:
        """Finalize the current word and add it to the sentence."""
        pattern = self.state.get_pattern()
        if not pattern:
            return
        
        print(f"\n🔍 Pattern: {pattern}")
        suggestions = self.lookup_suggestions(pattern)
        
        if suggestions:
            # Auto-select top suggestion
            chosen_word = suggestions[0].word
            self.state.final_sentence.append(chosen_word)
            print(f"\n✅ Selected: '{chosen_word}'")
            print(f"📄 Sentence: {' '.join(self.state.final_sentence)}")
        
        self.state.clear_buffer()
        self.state.suggestions = suggestions
    
    def watch_inactivity(self) -> None:
        """Background thread to monitor inactivity and finalize words."""
        while self.running:
            time.sleep(0.1)
            if self.state.is_inactive(INACTIVITY_TIMEOUT):
                self.finalize_word()
    
    def on_key_press(self, event) -> None:
        """Handle keyboard input."""
        key = event.name
        
        # Ignore multi-character keys (like 'shift', 'ctrl', etc.)
        if len(key) > 1:
            return
        
        # Check if it's a number key (testing mode)
        finger_code = NUMBER_TO_FINGER.get(key)
        
        if finger_code:
            self.state.add_finger(finger_code)
            print(f"👆 Finger: {finger_code} | Buffer: {self.state.buffer}")
    
    def run(self) -> None:
        """Start the gesture typing system."""
        print("=" * 70)
        print("🎹 GESTURE TYPING SYSTEM - CONTEXT-AWARE MODE")
        print("=" * 70)
        print("\nNumber Key Mapping:")
        print("  1=Left Pinky  2=Left Ring  3=Left Middle  4=Left Index  5=Left Thumb")
        print("  6=Right Thumb 7=Right Index 8=Right Middle 9=Right Ring 0=Right Pinky")
        print(f"\nInactivity timeout: {INACTIVITY_TIMEOUT}s")
        
        if self.lm.use_kenlm:
            print("🧠 Language Model: ENABLED (context-aware ranking)")
        else:
            print("📊 Language Model: DISABLED (frequency-based ranking)")
        
        print("\nType numbers to gesture type. Press Ctrl+C to exit.\n")
        print("=" * 70)
        
        # Start inactivity watcher
        watcher_thread = threading.Thread(target=self.watch_inactivity, daemon=True)
        watcher_thread.start()
        
        # Register keyboard handler
        keyboard.on_press(self.on_key_press)
        
        try:
            keyboard.wait()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            self.running = False
            self.db.close()

# ============ ENTRY POINT ============
if __name__ == "__main__":
    import sys
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print("❌ Database not found!")
        print("📦 Run: python database_builder.py")
        sys.exit(1)
    
    system = GestureTypingSystem()
    system.run()