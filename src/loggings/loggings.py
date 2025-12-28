from src.database.vector_db_interface import VectorDBInterface
import json
import numpy as np


def log(msg):
    print(msg)


def log_vect_db(vec_db: VectorDBInterface, log_file: str):
    with open(log_file, 'w', encoding='utf-8') as f:
        for vec in vec_db.get_all_chunks():
            f.write(vec.txt + '\n')

            # Convert embedding to string
            if isinstance(vec.embedding, np.ndarray):
                f.write(json.dumps(vec.embedding.tolist()) + '\n')
            else:
                f.write(str(vec.embedding) + '\n')

            # Convert metadata dict to JSON string
            f.write(json.dumps(vec.metadata, ensure_ascii=False) + '\n')
            f.write("----------------\n")
