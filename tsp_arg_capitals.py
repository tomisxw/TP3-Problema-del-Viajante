import pandas as pd
import numpy as np
import random

def build_distance_matrix(df):
    if df.shape[0] == df.shape[1]-1:
        cities = df.iloc[:,0].astype(str).tolist()
        D = df.iloc[:,1:].to_numpy(dtype=float)
        return cities, D

    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if num_cols:
        dist_col = num_cols[-1]
        non_num = [c for c in df.columns if c != dist_col]
        if len(non_num) >= 2:
            a_col, b_col = non_num[0], non_num[1]
            df2 = df.copy()
            df2[a_col] = df2[a_col].astype(str).str.strip()
            df2[b_col] = df2[b_col].astype(str).str.strip()
            cities = sorted(set(df2[a_col]).union(set(df2[b_col])))
            idx = {c:i for i,c in enumerate(cities)}
            n = len(cities)
            D = np.full((n,n), np.inf, dtype=float)
            np.fill_diagonal(D, 0.0)
            for _, row in df2.iterrows():
                i = idx[row[a_col]]
                j = idx[row[b_col]]
                d = float(row[dist_col])
                D[i,j] = d
                D[j,i] = d
            finite = D[np.isfinite(D)]
            if np.isinf(D).any():
                fallback = float(np.nanmax(finite)) if finite.size else 1e6
                D[np.isinf(D)] = fallback
            return cities, D

    n = df.shape[0]
    cities = [f"Ciudad_{i+1}" for i in range(n)]
    D = df.to_numpy(dtype=float)
    return cities, D

def tour_length(order, D):
    total = 0.0
    for i in range(len(order)-1):
        total += D[order[i], order[i+1]]
    total += D[order[-1], order[0]]
    return total

def nearest_neighbor(start, D):
    n = D.shape[0]
    unvisited = set(range(n))
    order = [start]
    unvisited.remove(start)
    while unvisited:
        last = order[-1]
        next_city = min(unvisited, key=lambda j: D[last, j])
        order.append(next_city)
        unvisited.remove(next_city)
    return order, tour_length(order, D)

def best_nearest_neighbor_all(D):
    best_order, best_len = None, float("inf")
    for s in range(D.shape[0]):
        o, L = nearest_neighbor(s, D)
        if L < best_len:
            best_len, best_order = L, o
    return best_order, best_len

def cyclic_crossover(p1, p2):
    n = len(p1)
    child = [None]*n
    index = 0
    while child[index] is None:
        child[index] = p1[index]
        index = p2.index(p1[index])
    for i in range(n):
        if child[i] is None:
            child[i] = p2[i]
    return child

def swap_mutation(perm):
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]

def tournament_select(pop, fitness, k=3):
    best_idx = None
    for _ in range(k):
        idx = random.randrange(len(pop))
        if best_idx is None or fitness[idx] < fitness[best_idx]:
            best_idx = idx
    return pop[best_idx]

def run_ga(D, N=50, M=200, pc=0.9, pm=0.2, elitism=2, seed=42):
    random.seed(seed)
    n = D.shape[0]
    base = list(range(n))
    population = [random.sample(base, n) for _ in range(N)]

    def fit(order): return tour_length(order, D)

    for gen in range(M):
        fitness = [fit(ind) for ind in population]
        elite_indices = np.argsort(fitness)[:elitism]
        elites = [population[i][:] for i in elite_indices]

        new_pop = elites.copy()
        while len(new_pop) < N:
            p1 = tournament_select(population, fitness, k=3)
            p2 = tournament_select(population, fitness, k=3)
            if random.random() < pc:
                c1 = cyclic_crossover(p1, p2)
                c2 = cyclic_crossover(p2, p1)
            else:
                c1, c2 = p1[:], p2[:]
            if random.random() < pm:
                swap_mutation(c1)
            if random.random() < pm and len(new_pop)+1 < N:
                swap_mutation(c2)
            new_pop.append(c1)
            if len(new_pop) < N:
                new_pop.append(c2)
        population = new_pop[:N]

    fitness = [fit(ind) for ind in population]
    best_idx = int(np.argmin(fitness))
    return population[best_idx], fitness[best_idx]

def mostrar_menu():
    print("\n" + "="*70)
    print("    PROBLEMA DEL VIAJANTE - CAPITALES DE ARGENTINA")
    print("="*70)
    print("1. Ruta desde una capital específica (Heurística Vecino Más Cercano)")
    print("2. Mejor ruta global (Heurística desde todas las capitales)")
    print("3. Ruta óptima (Algoritmo Genético)")
    print("4. Salir")
    print("="*70)

def mostrar_capitales(cities):
    print("\nCapitales disponibles:")
    print("-" * 50)
    for i, ciudad in enumerate(cities, 1):
        print(f"{i:2d}. {ciudad}")
    print("-" * 50)

def opcion_a(cities, D):
    """Opción A: Ruta desde una capital específica"""
    print("\n" + "="*50)
    print("OPCIÓN A: RUTA DESDE CAPITAL ESPECÍFICA")
    print("="*50)
    
    mostrar_capitales(cities)
    
    while True:
        try:
            opcion = input(f"\nSeleccione una capital (1-{len(cities)}): ").strip()
            if opcion.lower() == 'salir':
                return
            
            idx = int(opcion) - 1
            if 0 <= idx < len(cities):
                start_city = cities[idx]
                break
            else:
                print(f"Por favor, ingrese un número entre 1 y {len(cities)}")
        except ValueError:
            print("Por favor, ingrese un número válido o 'salir' para volver al menú")
    
    # Ejecutar heurística desde la capital seleccionada
    start_idx = cities.index(start_city)
    order, length = nearest_neighbor(start_idx, D)
    
    print(f"\nRESULTADO - HEURÍSTICA VECINO MÁS CERCANO")
    print(f"Ciudad de partida: {start_city}")
    print(f"Longitud total del trayecto: {length:.2f} km")
    print(f"Recorrido completo:")
    
    recorrido = [cities[i] for i in order]
    print(f"   {start_city}", end="")
    for ciudad in recorrido[1:]:
        print(f" → {ciudad}", end="")
    print(f" → {start_city}")
    
    input("\nPresione Enter para continuar...")

def opcion_b(cities, D):
    """Opción B: Mejor ruta global probando todas las capitales"""
    print("\n" + "="*50)
    print("OPCIÓN B: MEJOR RUTA GLOBAL")
    print("="*50)
    print("Probando todas las capitales como punto de inicio...")
    
    best_order, best_length = best_nearest_neighbor_all(D)
    start_city = cities[best_order[0]]
    
    print(f"\nMEJOR RESULTADO ENCONTRADO")
    print(f"Mejor ciudad de partida: {start_city}")
    print(f"Longitud mínima del trayecto: {best_length:.2f} km")
    print(f"Recorrido óptimo:")
    
    recorrido = [cities[i] for i in best_order]
    print(f"   {start_city}", end="")
    for ciudad in recorrido[1:]:
        print(f" → {ciudad}", end="")
    print(f" → {start_city}")
    
    input("\nPresione Enter para continuar...")

def opcion_c(cities, D):
    """Opción C: Algoritmo Genético"""
    print("\n" + "="*50)
    print("OPCIÓN C: ALGORITMO GENÉTICO")
    print("="*50)
    
    # Parámetros del AG
    print("Configuración del Algoritmo Genético:")
    print("- Población: 100 individuos")
    print("- Generaciones: 500")
    print("- Probabilidad de cruce: 0.9")
    print("- Probabilidad de mutación: 0.2")
    print("- Elitismo: 5 mejores individuos")
    
    print("\nEjecutando Algoritmo Genético... (esto puede tomar unos segundos)")
    
    # Ejecutar AG con parámetros optimizados
    best_order, best_length = run_ga(D, N=100, M=500, pc=0.9, pm=0.2, elitism=5, seed=42)
    start_city = cities[best_order[0]]
    
    print(f"\nRESULTADO - ALGORITMO GENÉTICO")
    print(f"Ciudad de partida: {start_city}")
    print(f"Longitud óptima del trayecto: {best_length:.2f} km")
    print(f"Recorrido óptimo:")
    
    recorrido = [cities[i] for i in best_order]
    print(f"   {start_city}", end="")
    for ciudad in recorrido[1:]:
        print(f" → {ciudad}", end="")
    print(f" → {start_city}")
    
    input("\nPresione Enter para continuar...")

def main():
    csv_file = "TablaCapitales.csv"
    
    try:
        df = pd.read_csv(csv_file)
        cities, D = build_distance_matrix(df)
        print(f"Datos cargados correctamente: {len(cities)} capitales")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{csv_file}'")
        print("Por favor, asegúrese de que el archivo CSV esté en el directorio actual.")
        return
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return
    
    # Menú principal
    while True:
        mostrar_menu()
        
        try:
            opcion = input("Seleccione una opción (1-4): ").strip()
            
            if opcion == "1":
                opcion_a(cities, D)
            elif opcion == "2":
                opcion_b(cities, D)
            elif opcion == "3":
                opcion_c(cities, D)
            elif opcion == "4":
                print("\n¡Gracias por usar el programa! 👋")
                break
            else:
                print("\nOpción no válida. Por favor, seleccione 1, 2, 3 o 4.")
                input("Presione Enter para continuar...")
                
        except KeyboardInterrupt:
            print("\n\n¡Programa interrumpido por el usuario! 👋")
            break
        except Exception as e:
            print(f"\nError inesperado: {e}")
            input("Presione Enter para continuar...")

if __name__ == "__main__":
    main()
