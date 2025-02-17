from typing import List
class Graph:
    def __init__(self, nodes: List[str], edges: List[str]):
        self.graph = dict()
        self.nodes = nodes
        self.edges = edges
        for node in nodes:
            self.graph[node] = {
                "child": [],
                "parent": []
            }
        for edge in edges:
            p, c = edge.split('->')
            self.graph[p]["child"].append(c)
            self.graph[c]["parent"].append(p)

    def __str__(self):
        return 'edges: {}, nodes: {}'.format(self.edges, self.nodes)

    def isCause(self, X, Y):
        # return True if X is a cause of Y
        assert X in self.nodes and Y in self.nodes
        if Y in self.graph[X]["child"]:
            return True
        for Z in self.graph[X]["child"]:
            if self.isCause(Z, Y):
                return True
        return False

    def subgraph(self):
        subgraphs = []
        has_parrent_nodes = []
        for node in self.graph:
            if self.graph[node]["parent"]:
                edges = [parent + '->' + node for parent in self.graph[node]["parent"]]
                nodes = [node] + self.graph[node]["parent"]
                subgraphs.append(Graph(nodes, edges))
                has_parrent_nodes.append(node)
        if len(has_parrent_nodes) > 1:
            subsets = [[]]
            for node in has_parrent_nodes:
                subsets += [subset + [node] for subset in subsets]
            subsets = [subset for subset in subsets if len(subset) >= 2]
            for subset in subsets:
                edges = []
                nodes = set()
                for node in subset:
                    edges += [parent + '->' + node for parent in self.graph[node]["parent"]]
                    nodes = nodes.union(set([node] + self.graph[node]["parent"]))
                subgraphs.append(Graph(list(nodes), edges))
        return subgraphs