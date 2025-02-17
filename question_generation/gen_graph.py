import networkx as nx
import tqdm
import os
import argparse
import random


def generate_all_graph(n, max_edges):
    """
    generate all non-isomorphic DAGs given n vertices
    """
    nodes = list(range(n))
    edge_list = list()
    for i in range(n - 1):
        for j in range(i + 1, n):
            edge_list.append((i, j))
    
    edge_size = len(edge_list)
    assert edge_size == n * (n - 1) // 2
    assert max_edges <= 0 or max_edges >= n - 1

    graph_list = []
    for k in tqdm.tqdm(range(2 ** edge_size)):
        edges = []
        for i in range(edge_size):
            if (k >> i) % 2 == 1:
                edges.append(edge_list[i])
        if max_edges > 0 and len(edges) > max_edges: continue
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        if nx.is_weakly_connected(G):
            flag = True
            for g in graph_list:
                # if nx.is_isomorphic(g, G):
                if nx.vf2pp_is_isomorphic(g, G):
                    flag = False
                    break
            if flag:
                graph_list.append(G)

    return graph_list

def generate_some_graph(n, size, max_try, max_edges):
    nodes = list(range(n))
    edge_list = list()
    for i in range(n - 1):
        for j in range(i + 1, n):
            edge_list.append((i, j))
    edge_size = len(edge_list)
    assert max_edges <= 0 or max_edges >= n - 1

    graph_list = []
    for s in tqdm.tqdm(range(size)):
        count = 0
        find_flag = False
        while count < max_try:
            count += 1
            if max_edges > 0:
                edge_num = random.randint(n-1, max_edges)
            else:
                edge_num = random.randint(n-1, edge_size)  # 连通图，至少要有n-1条边
            edge_index = random.sample(range(edge_size), edge_num)
            edges = []
            for index in edge_index:
                edges.append(edge_list[index])
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            if nx.is_weakly_connected(G):
                flag = True
                for g in graph_list:
                    if nx.vf2pp_is_isomorphic(g, G):
                        flag = False
                        break
                if flag:
                    graph_list.append(G)
                    find_flag = True
                    break
        if not find_flag:
            print("Failed to find the {:d}th DAG".format(s))
    return graph_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node_num",
        "-n",
        type=int,
        nargs="+",
        help="Number of nodes of the generating DAGs. If 2 numbers are provided, we regard it as the range of node num [num_min, num_max].",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        help="The output txt file",
    )
    parser.add_argument(
        "--graph_num",
        "-g",
        type=int,
        default=0,
        help="Number of graphs (for a certain size), zero or negative number represents generating all possible DAGs",
    )
    parser.add_argument(
        "--max_try",
        type=int,
        default=10,
        help="max trying number when finding a single DAG",
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=-1,
        help="maximum number of edges"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    dirname = os.path.dirname(args.output_file)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    
    graph_list = []
    if len(args.node_num) == 1:
        n_min, n_max = args.node_num[0], args.node_num[0]
    else:
        n_min, n_max = args.node_num[0], args.node_num[1]

    for n in range(n_min, n_max + 1):
        if args.graph_num <= 0:
            tmp = generate_all_graph(n, args.max_edges)
        else:
            tmp = generate_some_graph(n, args.graph_num, args.max_try, args.max_edges)
        graph_list += tmp
        print("Generation finish: {:d} DAGs of {:d} vertices".format(len(tmp), n))
    f_out = open(args.output_file, 'w')
    for graph in graph_list:
        print(graph.edges, file=f_out)
    f_out.close()


