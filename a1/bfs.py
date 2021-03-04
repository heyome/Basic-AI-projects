MY_GRAPH = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'X'],
    'C': ['A', 'D', 'Y'],
    'D': ['B', 'C', 'Z'],
    'X': ['B'],
    'Y': ['C'],
    'Z': ['D']
}

def bfs(start, end, graph):
    q = [start]
    closed_list = set()  # places we don't want to revisit
    parents = {start: 'START'} # lookup table for how we got here
    while q:  # shorthand for "not empty"
        to_explore = q.pop(0)
        if to_explore in closed_list:
            continue
        if to_explore == end:
            return path_to_here(to_explore, parents)
        closed_list.add(to_explore)
        neighbors = graph[to_explore]
        for n in neighbors:
            if not n in parents:
                parents[n] = to_explore
        q.extend(neighbors)  # append would add the list as one item
    return None
        
def path_to_here(node, parents):
    path = []
    while node != 'START':
        path.insert(0, node)
        node = parents[node]
    return path
