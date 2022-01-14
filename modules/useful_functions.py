import pickle
import os

def load_or_create_cache(n_reviews, things_to_store_name, function_to_compute_things_to_store, *args):
    cache_pkl = f'caches/{n_reviews}/{things_to_store_name}.pkl'
    try:
        with open(cache_pkl, 'rb') as f:
            things_to_store = pickle.load(f)
        print(f'load {things_to_store_name} from {cache_pkl}\n')
    except IOError:
        print(f'{cache_pkl} not found, start computing {things_to_store_name}...')
        things_to_store = function_to_compute_things_to_store(*args)
        try:
            os.mkdir('/'.join(cache_pkl.split('/')[:-1]))
            os.mkdir(f'fig/{n_reviews}')
            os.mkdir(f'tests/fig/{n_reviews}')
        except OSError:
            pass
        with open(cache_pkl, 'wb') as f:
            pickle.dump(things_to_store, f)
    return things_to_store
