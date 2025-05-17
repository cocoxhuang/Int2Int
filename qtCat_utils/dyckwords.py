from sage.all import DyckWords      # All of .area(), .dinv(), .bounce(), and .area_dinv_to_bounce_area_map() come from the sage.combinat.dyck_word module

import time

def area_data(n, filename):

    '''
    n : int, the length of Dyck words
    filename : str, the name of the file to save the data
    '''

    # time how long it takes to run the code
    start = time.time()

    vecs = DyckWords(n)
    print(f"Retrieving Dyck words of semilength {n}...")

    vectors = [[list(entry), entry.area()] for entry in vecs]
    # Write list to file
    with open(filename, 'w') as file:
        file.write('\n'.join(map(str, vectors)))
    print(f"Success. Dyck words of length {n} and their areas are saved in {filename}.")
    end = time.time()
    print(f"Time taken to generate data: {time.time() - start} seconds.")

def dinv_data(n, filename):
    '''
    n : int, the length of Dyck words
    filename : str, the name of the file to save the data
    '''

    start = time.time()
    vecs = DyckWords(n)
    print(f"Retrieving Dyck words of semilength {n}...")

    vectors = [[list(entry), entry.dinv()] for entry in vecs]
    # Write list to file
    with open(filename, 'w') as file:
        file.write('\n'.join(map(str, vectors)))
    print(f"Success. Dyck words of length {n} and their dinv are saved in {filename}.")
    print(f"Time taken to generate data: {time.time() - start} seconds.")

def bounce_data(n, filename):
    '''
    n : int, the length of Dyck words
    filename : str, the name of the file to save the data
    '''

    start = time.time()
    vecs = DyckWords(n)
    print(f"Retrieving Dyck words of semilength {n}...")

    vectors = [[list(entry), entry.bounce()] for entry in vecs]
    # Write list to file
    with open(filename, 'w') as file:
        file.write('\n'.join(map(str, vectors)))
    print(f"Success. Dyck words of length {n} and their bounce are saved in {filename}.")
    print(f"Time taken to generate data: {time.time() - start} seconds.")

def area_dinv_data(n, filename):
    '''
    n : int, the length of Dyck words
    filename : str, the name of the file to save the data
    '''

    start = time.time()
    vecs = DyckWords(n)
    print(f"Retrieving Dyck words of semilength {n}...")

    vectors = [[list(entry), entry.area(), entry.dinv()] for entry in vecs]
    # Write list to file
    with open(filename, 'w') as file:
        file.write('\n'.join(map(str, vectors)))
    print(f"Success. Dyck words of length {n} and their area,dinv are saved in {filename}.")
    print(f"Time taken to generate data: {time.time() - start} seconds.")

def bounce_area_data(n, filename):
    '''
    n : int, the length of Dyck words
    filename : str, the name of the file to save the data
    '''

    start = time.time()
    vecs = DyckWords(n)
    print(f"Retrieving Dyck words of semilength {n}...")

    vectors = [[list(entry), entry.bounce(), entry.area()] for entry in vecs]
    # Write list to file
    with open(filename, 'w') as file:
        file.write('\n'.join(map(str, vectors)))
    print(f"Success. Dyck words of length {n} and their bounce,area are saved in {filename}.")
    print(f"Time taken to generate data: {time.time() - start} seconds.")

def area_dinv_bounce_data(n, filename):
    '''
    n : int, the length of Dyck words
    filename : str, the name of the file to save the data
    '''
    start = time.time()
    vecs = DyckWords(n)
    print(f"Retrieving Dyck words of semilength {n}...")

    vectors = [[list(entry), entry.area(), entry.dinv(), entry.bounce()] for entry in vecs]
    # Write list to file
    with open(filename, 'w') as file:
        file.write('\n'.join(map(str, vectors)))
    print(f"Success. Dyck words of length {n} and their area,dinv,bounce are saved in {filename}.")
    print(f"Time taken to generate data: {time.time() - start} seconds.")

def zeta_map_data(n, filename):
    '''
    n : int, the length of Dyck words
    filename : str, the name of the file to save the data
    '''

    start = time.time()
    vecs = DyckWords(n)
    print(f"Retrieving Dyck words of semilength {n}...")

    # zeta map
    vectors = [[list(entry), entry.area_dinv_to_bounce_area_map()] for entry in vecs]
    # Write list to file
    with open(filename, 'w') as file:
        file.write('\n'.join(map(str, vectors)))
    print(f"Success. Dyck words of length {n} and their area,dinv,bounce are saved in {filename}.")
    print(f"Time taken to generate data: {time.time() - start} seconds.")