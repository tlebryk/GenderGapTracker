from pathlib import Path
import os
# os.path.join(os.path.dirname(__file__)
os.path.join(Path(__file__), "rules\author_blocklist.txt")
config = {
    'MONGO_ARGS': {
        'host': ['mongo0', 'mongo1', 'mongo2'],
        'port': 27017,
        # 'username': USERNAME,
        # 'password': PASSWORD,    
        'authSource': 'admin',
        'readPreference': 'nearest'
    },
    'GENDER_RECOGNITION': {
        'GENDERIZE_ENABLED': False,
        'GENDERAPI_ENABLED': True,
        # 'GENDERAPI_TOKEN': XXXXXXXXXX,
        'HOST': 'localhost',
        'PORT': 5000
    },
    'NLP': {
        'MAX_BODY_LENGTH': 20000,
        'AUTHOR_BLOCKLIST': os.path.join(os.path.dirname(Path(__file__)), r"rules/author_blocklist.txt"),
        'NAME_PATTERNS': os.path.join(os.path.dirname(Path(__file__)), r'rules/name_patterns.jsonl'),
        'QUOTE_VERBS': os.path.join(os.path.dirname(Path(__file__)), r'rules/quote_verb_list.txt')
    }
}