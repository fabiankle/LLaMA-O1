import networkx as nx
import matplotlib.pyplot as plt


def add_nodes_edges(tree_node, graph):
    """Recursively add nodes and edges to the graph."""
    if tree_node is None:
        return

    # Add the node with its attributes
    graph.add_node(tree_node.index, state=tree_node.state, value=tree_node.value, meta=tree_node.meta)

    # Add edges to children
    for child in tree_node.children:
        graph.add_edge(tree_node.index, child.index)
        add_nodes_edges(child, graph)


def visualize_tree(root):
    """Visualize the tree using networkx and matplotlib."""
    graph = nx.DiGraph()
    add_nodes_edges(root, graph)

    # Use pygraphviz layout if available
    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    except ImportError:
        pos = nx.spring_layout(graph)

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=False, arrows=True, node_size=3000, node_color='lightblue')

    # Draw node labels with selected attributes
    node_labels = {node: f"{data['state']}\nValue: {data['value']}\nMeta: {data['meta']}"
                   for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_color='black')

    plt.title("Tree Visualization")
    plt.axis('off')
    plt.show()

# Example usage:
# Assuming you have a tree structure created with TreeNode
# root = TreeNode("root")
# child1 = TreeNode("child1", parent=root)
# child2 = TreeNode("child2", parent=root)
# root.children.extend([child1, child2])
# visualize_tree(root)