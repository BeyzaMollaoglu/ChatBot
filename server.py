#!/usr/bin/env python3
"""
Enhanced AI Search Server - Python Version
Supports Turkish text processing, HTML/TXT file indexing, and semantic search
"""

import os
import re
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Web framework
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# HTML parsing
from bs4 import BeautifulSoup

# Vector search and embeddings
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    file_type: str
    similarity: float = 0.0
    matched_words: List[str] = None
    
    def __post_init__(self):
        if self.matched_words is None:
            self.matched_words = []

class TurkishTextProcessor:
    """Turkish text normalization and processing utilities"""
    
    # Turkish stopwords
    STOP_WORDS = {
        "ve", "veya", "ile", "de", "da", "mi", "mı", "mu", "mü", "ya", "yada", "ki",
        "bir", "bu", "şu", "o", "şey", "çok", "az", "daha", "en", "ama", "fakat", "ancak",
        "ben", "sen", "o", "biz", "siz", "onlar", "var", "yok", "icin", "için", "gibi",
        "istiyorum", "istemek", "gitmek", "gidebilir", "yapmak", "lütfen", "sayfasi", "sayfası",
        "sayfaya", "sayfasina", "sayfasına", "sayfa", "git", "aç", "ac", "açar", "açmak"
    }
    
    # Turkish suffixes for basic stemming
    SUFFIXES = [
        "den", "dan", "ten", "tan",
        "nin", "nın", "nun", "nün", "in", "ın", "un", "ün",
        "de", "da", "te", "ta",
        "ye", "ya",
        "yı", "yi", "yu", "yü", "i", "ı", "u", "ü",
        "e", "a",
        "ne", "na"
    ]
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize Turkish text by converting special characters"""
        if not text:
            return ""
        
        text = text.lower()
        # Turkish character mappings
        replacements = {
            "ı": "i", "İ": "i",
            "ğ": "g", "Ğ": "g",
            "ü": "u", "Ü": "u",
            "ş": "s", "Ş": "s",
            "ö": "o", "Ö": "o",
            "ç": "c", "Ç": "c"
        }
        
        for tr_char, en_char in replacements.items():
            text = text.replace(tr_char, en_char)
        
        # Remove diacritics
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        return text
    
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize Turkish text, preserving Turkish letters"""
        if not text:
            return []
        
        # Normalize and replace non-letter/non-digit/non-space with space
        normalized = cls.normalize(text)
        # Keep Unicode letters and numbers, replace everything else with space
        cleaned = re.sub(r'[^\p{L}\p{N}\s]', ' ', normalized, flags=re.UNICODE)
        
        # Split and filter empty tokens
        tokens = [token for token in cleaned.split() if token]
        return tokens
    
    @classmethod
    def strip_suffix(cls, word: str) -> str:
        """Basic Turkish suffix stripping"""
        if not word or len(word) < 3:
            return word
        
        for suffix in cls.SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        
        return word
    
    @classmethod
    def extract_search_keys(cls, query: str) -> List[str]:
        """Extract meaningful search keywords from query"""
        tokens = cls.tokenize(query)
        keys = []
        
        for token in tokens:
            stemmed = cls.strip_suffix(token)
            if len(stemmed) >= 2 and stemmed not in cls.STOP_WORDS:
                keys.append(stemmed)
        
        return keys[:10]  # Limit to 10 keywords

class ContentExtractor:
    """Extract and process content from various file types"""
    
    SUPPORTED_EXTENSIONS = {'.html', '.htm', '.txt'}
    EXCLUDE_DIRS = {'node_modules', '.git', 'assets', 'images', 'img', 'css', 'js'}
    
    @classmethod
    def find_supported_files(cls, directory: Path) -> List[Path]:
        """Recursively find supported files"""
        files = []
        
        def scan_dir(dir_path: Path):
            try:
                for item in dir_path.iterdir():
                    if item.is_dir():
                        if (item.name not in cls.EXCLUDE_DIRS and 
                            not item.name.startswith('.')):
                            scan_dir(item)
                    elif item.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                        files.append(item)
            except PermissionError:
                print(f"Warning: Permission denied accessing {dir_path}")
            except Exception as e:
                print(f"Warning: Error scanning {dir_path}: {e}")
        
        scan_dir(directory)
        return files
    
    @staticmethod
    def slugify(text: str) -> str:
        """Generate URL-friendly slug"""
        if not text:
            return ""
        
        # Normalize and remove diacritics
        normalized = unicodedata.normalize('NFD', text.lower())
        ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # Keep only alphanumeric and spaces/hyphens
        cleaned = re.sub(r'[^\w\s-]', '', ascii_text)
        slug = re.sub(r'\s+', '-', cleaned.strip())
        
        return slug[:64]  # Limit length
    
    @staticmethod
    def clean_heading_title(title: str) -> str:
        """Clean heading titles by removing (h1), (h2) etc."""
        if not title:
            return ""
        return re.sub(r'\s*\(h[1-6]\)\s*$', '', title, flags=re.IGNORECASE).strip()
    
    @classmethod
    def extract_from_txt(cls, file_path: Path, public_dir: Path) -> List[SearchResult]:
        """Extract sections from TXT files"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if not content.strip():
                return []
            
            rel_path = file_path.relative_to(public_dir).as_posix()
            file_name = file_path.stem
            
            lines = [line for line in content.split('\n') if line.strip()]
            sections = []
            current_section = ''
            current_title = file_name
            
            for i, line in enumerate(lines):
                # Detect likely headers
                is_header = (
                    len(line) < 100 and (
                        re.match(r'^[A-Z][^.!?]*$', line) or
                        re.match(r'^\d+\.?\s', line) or
                        re.match(r'^[-=]{3,}', line) or
                        line.isupper()
                    )
                )
                
                if is_header and len(current_section) > 100:
                    # Save previous section
                    sections.append(SearchResult(
                        title=current_title,
                        url=f"/{rel_path}#section-{len(sections) + 1}",
                        content=current_section.strip(),
                        file_type='txt'
                    ))
                    
                    current_title = line if len(line) < 50 else file_name
                    current_section = ''
                else:
                    current_section += line + '\n'
            
            # Add final section
            if current_section.strip():
                sections.append(SearchResult(
                    title=current_title,
                    url=f"/{rel_path}#section-{len(sections) + 1}",
                    content=current_section.strip()[:4000],
                    file_type='txt'
                ))
            
            # If no sections found, treat as single document
            if not sections:
                sections.append(SearchResult(
                    title=file_name,
                    url=f"/{rel_path}",
                    content=content.strip()[:4000],
                    file_type='txt'
                ))
            
            return sections
            
        except Exception as e:
            print(f"Error processing TXT {file_path}: {e}")
            return []
    
    @classmethod
    def extract_from_html(cls, file_path: Path, public_dir: Path) -> List[SearchResult]:
        """Extract sections from HTML files"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')
            
            rel_path = file_path.relative_to(public_dir).as_posix()
            file_name = file_path.stem
            
            # Extract page metadata
            page_title = soup.find('title')
            page_title = page_title.get_text().strip() if page_title else file_name
            
            meta_desc = soup.find('meta', {'name': 'description'})
            meta_description = meta_desc.get('content', '') if meta_desc else ''
            
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            # If no headings, treat as single page
            if not headings:
                body = soup.find('body')
                body_text = body.get_text() if body else soup.get_text()
                body_text = re.sub(r'\s+', ' ', body_text).strip()
                
                if not body_text:
                    return []
                
                full_content = f"{meta_description}\n{body_text}"
                return [SearchResult(
                    title=page_title,
                    url=f"/{rel_path}",
                    content=full_content[:4000],
                    file_type='html'
                )]
            
            sections = []
            
            # Add page overview section
            body = soup.find('body')
            if body:
                first_content = body.find()
                if first_content:
                    overview = re.sub(r'\s+', ' ', first_content.get_text()).strip()
                    if overview and len(overview) > 50:
                        sections.append(SearchResult(
                            title=f"{page_title} (Overview)",
                            url=f"/{rel_path}",
                            content=f"{meta_description}\n{overview}"[:4000],
                            file_type='html'
                        ))
            
            # Process each heading section
            for idx, heading in enumerate(headings):
                raw_title = heading.get_text().strip() or f"Section {idx + 1}"
                title = cls.clean_heading_title(raw_title)
                
                heading_id = heading.get('id')
                if not heading_id:
                    heading_id = cls.slugify(title)
                
                # Get content until next heading of same or higher level
                current_level = int(heading.name[1])
                content = heading.get_text()
                
                for sibling in heading.find_next_siblings():
                    if sibling.name and re.match(r'^h[1-6]$', sibling.name):
                        sibling_level = int(sibling.name[1])
                        if sibling_level <= current_level:
                            break
                    content += "\n" + sibling.get_text()
                
                content = re.sub(r'\s+', ' ', content).strip()
                if not content or len(content) < 20:
                    continue
                
                sections.append(SearchResult(
                    title=f"{title} ({heading.name})",
                    url=f"/{rel_path}#{heading_id}",
                    content=content[:4000],
                    file_type='html'
                ))
            
            return sections
            
        except Exception as e:
            print(f"Error processing HTML {file_path}: {e}")
            return []
    
    @classmethod
    def extract_content_from_file(cls, file_path: Path, public_dir: Path) -> List[SearchResult]:
        """Extract content from a file based on its extension"""
        ext = file_path.suffix.lower()
        
        if ext == '.txt':
            return cls.extract_from_txt(file_path, public_dir)
        elif ext in {'.html', '.htm'}:
            return cls.extract_from_html(file_path, public_dir)
        else:
            return []

class EnhancedSearchEngine:
    """AI-powered search engine with Turkish support"""
    
    def __init__(self, public_dir: Path):
        self.public_dir = public_dir
        self.text_processor = TurkishTextProcessor()
        self.content_extractor = ContentExtractor()
        
        # Search parameters
        self.min_similarity = float(os.getenv('SEARCH_MIN_SIM', '0.25'))
        self.diversity_gap = float(os.getenv('SEARCH_DIVERSITY_GAP', '0.35'))
        self.max_options = int(os.getenv('SEARCH_MAX_OPTIONS', '6'))
        
        # Initialize embeddings model
        print("🧠 Loading sentence transformer model...")
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Storage for documents and embeddings
        self.documents: List[SearchResult] = []
        self.embeddings: Optional[np.ndarray] = None
        self.stats = {
            'files': 0,
            'sections': 0,
            'timestamp': None,
            'file_types': []
        }
        
    def build_index(self):
        """Build the search index from files in public directory"""
        print("🌐 Building enhanced site search index...")
        
        files = self.content_extractor.find_supported_files(self.public_dir)
        print(f"📁 Found {len(files)} supported files")
        
        self.documents = []
        processed_files = 0
        file_types = set()
        
        for file_path in files:
            try:
                sections = self.content_extractor.extract_content_from_file(file_path, self.public_dir)
                
                for section in sections:
                    file_types.add(section.file_type)
                    self.documents.append(section)
                
                processed_files += 1
                
            except Exception as e:
                print(f"⚠️ Could not process {file_path}: {e}")
        
        if not self.documents:
            print("⚠️ No indexable content found in public/ directory.")
            self.embeddings = None
            self.stats = {
                'files': processed_files,
                'sections': 0,
                'timestamp': datetime.now().isoformat(),
                'file_types': []
            }
            return
        
        # Create embeddings
        print("🔗 Creating document embeddings...")
        texts = []
        for doc in self.documents:
            enhanced_content = f"{doc.title}\n\n{doc.content}"
            texts.append(enhanced_content)
        
        self.embeddings = self.embeddings_model.encode(texts)
        
        self.stats = {
            'files': processed_files,
            'sections': len(self.documents),
            'timestamp': datetime.now().isoformat(),
            'file_types': list(file_types)
        }
        
        print(f"✅ Enhanced search index ready: {self.stats['files']} files, {self.stats['sections']} sections")
        print(f"📊 File types: {', '.join(self.stats['file_types'])}")
    
    def calculate_word_similarity(self, query: str, document: SearchResult) -> float:
        """Calculate similarity based on word matching"""
        search_keys = self.text_processor.extract_search_keys(query)
        if not search_keys:
            return 0
        
        # Combine all searchable text
        title = self.text_processor.normalize(document.title)
        content = self.text_processor.normalize(document.content)
        url = self.text_processor.normalize(document.url)
        all_text = f"{title} {content} {url}"
        
        hits = 0
        strong_hits = 0
        matched_words = []
        
        for key in search_keys:
            if not key:
                continue
            if key in all_text:
                hits += 1
                matched_words.append(key)
                # Bonus for matches in title or URL
                if key in title or key in url:
                    strong_hits += 1
        
        if hits == 0:
            return 0
        
        base_score = hits / len(search_keys)
        bonus = strong_hits * 0.2
        final_score = min(base_score + bonus, 1.0)
        
        # Update matched words in document
        document.matched_words = matched_words
        
        return final_score
    
    def generate_preview(self, content: str, query: str, max_length: int = 150) -> str:
        """Generate content preview highlighting matching context"""
        search_keys = self.text_processor.extract_search_keys(query)
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if len(s.strip()) > 10]
        
        if not sentences:
            return content[:max_length] + ('...' if len(content) > max_length else '')
        
        # Find sentence with most word matches
        best_sentence = sentences[0]
        max_matches = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for key in search_keys if key in sentence_lower)
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence
        
        preview = best_sentence.strip()
        if len(preview) > max_length:
            preview = preview[:max_length] + '...'
        
        return preview
    
    def search(self, query: str) -> Dict:
        """Perform enhanced search with word-based matching"""
        if not self.documents or self.embeddings is None:
            return {
                'reply': "🔄 Arama indeksi hazırlanıyor. Lütfen birazdan tekrar deneyin.",
                'options': []
            }
        
        search_keys = self.text_processor.extract_search_keys(query)
        print(f"🔍 Searching for words: [{', '.join(search_keys)}] from query: \"{query}\"")
        
        if not search_keys:
            return {
                'reply': "Please use more specific words in your query so I can search better.",
                'suggestions': ["Try: 'sign in page'", "Try: 'new document'", "Try: 'home index'"],
                'options': []
            }
        
        # Calculate word-based similarities
        word_matches = []
        for doc in self.documents:
            word_sim = self.calculate_word_similarity(query, doc)
            if word_sim > 0:  # Only keep documents with word matches
                doc.similarity = word_sim
                word_matches.append(doc)
        
        # Sort by similarity
        word_matches.sort(key=lambda x: x.similarity, reverse=True)
        
        if not word_matches:
            return {
                'reply': "Uygun sonuç bulunamadı. Daha kısa ve belirgin kelimelerle yeniden dener misiniz? (ör. \"login\", \"kayıt\", \"panel\").",
                'options': []
            }
        
        # Group by page and select best match per page
        best_by_page = {}
        for doc in word_matches:
            url_base = doc.url.split('#')[0] if '#' in doc.url else doc.url
            if not url_base:
                continue
            
            if url_base not in best_by_page or doc.similarity > best_by_page[url_base].similarity:
                best_by_page[url_base] = doc
        
        # Get top results
        all_pages = sorted(best_by_page.values(), key=lambda x: x.similarity, reverse=True)
        top_pages = all_pages[:self.max_options]
        
        # Generate response options
        options = []
        seen_urls = set()
        
        for page in top_pages:
            target_url = page.url if '#' in page.url else page.url.rstrip('/')
            if target_url in seen_urls:
                continue
            seen_urls.add(target_url)
            
            # Generate clean label
            if '#' in page.url:
                label = self.content_extractor.clean_heading_title(page.title)
            else:
                label = target_url.lstrip('/').replace('.html', '').replace('.htm', '')
                if not label:
                    label = page.title
            
            options.append({
                'type': 'open_url',
                'url': target_url,
                'label': label,
                'preview': self.generate_preview(page.content, query)
            })
            
            if len(options) >= self.max_options:
                break
        
        # Generate reply
        if len(options) == 1:
            reply = "🎯 Sayfa bulundu. Buradan ulaşabilirsiniz:"
        else:
            reply = f"🎯 {len(options)} seçenek bulundu. Lütfen birini seçin:"
        
        # Collect statistics
        found_words = list(set(word for doc in top_pages for word in doc.matched_words))
        
        return {
            'reply': reply,
            'options': options,
            'search_query': query,
            'search_keys': search_keys,
            'found_words': found_words,
            'results_count': len(options)
        }

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize search engine - fix path resolution
script_dir = Path(__file__).parent
project_root = script_dir.parent
PUBLIC_DIR = project_root / "public"

print(f"📁 Script location: {script_dir}")
print(f"📁 Project root: {project_root}")
print(f"📁 Looking for public dir at: {PUBLIC_DIR}")
print(f"📁 Public dir exists: {PUBLIC_DIR.exists()}")

search_engine = EnhancedSearchEngine(PUBLIC_DIR)

# Build index on startup
search_engine.build_index()

@app.route('/')
def serve_index():
    """Serve static files from public directory"""
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(PUBLIC_DIR, filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Enhanced chat API with AI search"""
    try:
        data = request.get_json() or {}
        message = str(data.get('message', '')).strip()
        action = data.get('action')
        
        # Handle button actions
        if action and action.get('type') == 'open_url' and action.get('url'):
            return jsonify({
                'reply': f"Yönlendiriyorum: {action['url']}",
                'navigate': action['url']
            })
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Perform search
        result = search_engine.search(message)
        return jsonify(result)
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'error': 'Server error occurred while processing your request',
            'details': str(e) if os.getenv('FLASK_ENV') == 'development' else None
        }), 500

@app.route('/api/reindex-site', methods=['POST'])
def reindex_site():
    """Manual reindexing endpoint"""
    try:
        search_engine.build_index()
        return jsonify({'ok': True, 'stats': search_engine.stats})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/search-stats', methods=['GET'])
def search_stats():
    """Get search index statistics"""
    return jsonify({'stats': search_engine.stats})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'index_ready': len(search_engine.documents) > 0,
        'stats': search_engine.stats,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))  # Changed back to 3001 to match Node.js version
    
    print(f"🚀 Enhanced AI Search Server starting on http://localhost:{port}")
    print(f"📊 Index stats: {search_engine.stats}")
    
    # Development vs production settings
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    
    app.run(
        host='localhost',
        port=port,
        debug=debug_mode,
        threaded=True
    )