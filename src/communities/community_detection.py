import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
from networkx.algorithms.community import girvan_newman
from pathlib import Path
import matplotlib.pyplot as plt


PROCESSED_FILE = Path(
    "data/processed/college_msg_processed.csv"
)

OUTPUT_FILE = Path(
    "data/processed/louvain_communities.csv"
)

COMPARISON_FILE = Path(
    "results/tables/community_detection_comparison.csv"
)

COMPARISON_FIGURE = Path(
    "results/figures/community_detection_comparison.png"
)


def build_graph(df):
    """
    Build an undirected weighted graph.

    Nodes represent users.
    Edge weight represents the number of interactions
    between two users.
    """

    G = nx.Graph()

    for _, row in df.iterrows():

        source = int(row["source"])
        target = int(row["target"])

        if G.has_edge(source, target):

            G[source][target]["weight"] += 1

        else:

            G.add_edge(
                source,
                target,
                weight=1
            )

    return G


def detect_louvain(G):
    """
    Detect communities using Louvain.

    If the graph contains edge weights, they are used.
    """

    partition = community_louvain.best_partition(
        G,
        weight="weight",
        random_state=42
    )

    return partition


def detect_girvan_newman(
    G,
    max_levels=30
):
    """
    Run Girvan-Newman community detection.

    Edge betweenness is calculated using the unweighted
    network topology because CollegeMsg edge weights
    represent interaction frequency rather than distance.

    The partition with the highest modularity among
    the evaluated levels is returned.
    """

    def most_valuable_edge(graph):

        betweenness = (
            nx.edge_betweenness_centrality(
                graph
            )
        )

        return max(
            betweenness,
            key=betweenness.get
        )

    generator = girvan_newman(
        G,
        most_valuable_edge=most_valuable_edge
    )

    best_partition = None
    best_modularity = -1.0

    for level, partition in enumerate(
        generator,
        start=1
    ):

        if level > max_levels:
            break

        communities = tuple(
            frozenset(c)
            for c in partition
        )

        # Convert communities to a partition dictionary
        partition_dict = {
            node: community_id
            for community_id, community
            in enumerate(communities)
            for node in community
        }

        # Since G is unweighted, modularity is
        # calculated without edge weights.
        modularity = (
            community_louvain.modularity(
                partition_dict,
                G
            )
        )

        print(
            f"GN level {level}: "
            f"{len(communities)} communities, "
            f"modularity = {modularity:.6f}"
        )

        if modularity > best_modularity:

            best_modularity = modularity
            best_partition = communities

    return (
        best_partition,
        best_modularity
    )


def build_comparison_graph(
    G,
    sample_size=300
):
    """
    Create a connected subgraph for comparing
    Louvain and Girvan-Newman.

    The subgraph is obtained using BFS starting from
    the highest-degree node.

    A completely unweighted graph is returned so that
    both algorithms are evaluated under identical
    conditions.
    """

    # ------------------------------------------
    # LARGEST CONNECTED COMPONENT
    # ------------------------------------------

    largest_component = max(
        nx.connected_components(G),
        key=len
    )

    G_largest = G.subgraph(
        largest_component
    ).copy()

    # ------------------------------------------
    # SAMPLE SIZE
    # ------------------------------------------

    sample_size = min(
        sample_size,
        G_largest.number_of_nodes()
    )

    # ------------------------------------------
    # START FROM HIGHEST-DEGREE NODE
    # ------------------------------------------

    start_node = max(
        G_largest.degree(),
        key=lambda x: x[1]
    )[0]

    # ------------------------------------------
    # BFS EXPANSION
    # ------------------------------------------

    bfs_nodes = list(
        nx.bfs_tree(
            G_largest,
            start_node
        ).nodes()
    )

    nodes = bfs_nodes[:sample_size]

    # ------------------------------------------
    # ORIGINAL SAMPLE
    # ------------------------------------------

    G_sample = G_largest.subgraph(
        nodes
    ).copy()

    # ------------------------------------------
    # CREATE TRULY UNWEIGHTED GRAPH
    # ------------------------------------------

    G_comparison = nx.Graph()

    G_comparison.add_nodes_from(
        G_sample.nodes()
    )

    G_comparison.add_edges_from(
        G_sample.edges()
    )

    return G_sample, G_comparison


def run_community_comparison(G):
    """
    Compare Louvain and Girvan-Newman on the
    same 300-node unweighted subgraph.
    """

    print("\n" + "=" * 50)
    print("LOUVAIN VS GIRVAN-NEWMAN")
    print("=" * 50)

    # ==========================================
    # BUILD COMPARISON GRAPH
    # ==========================================

    G_sample, G_comparison = (
        build_comparison_graph(
            G,
            sample_size=300
        )
    )

    print("\nComparison graph:")
    print(
        "Nodes:",
        G_comparison.number_of_nodes()
    )

    print(
        "Edges:",
        G_comparison.number_of_edges()
    )

    # ==========================================
    # LOUVAIN
    # ==========================================

    print("\n----- Louvain -----")

    louvain_partition = detect_louvain(
        G_comparison
    )

    louvain_communities = len(
        set(
            louvain_partition.values()
        )
    )

    louvain_modularity = (
        community_louvain.modularity(
            louvain_partition,
            G_comparison
        )
    )

    print(
        "Communities:",
        louvain_communities
    )

    print(
        "Modularity:",
        louvain_modularity
    )

    # ==========================================
    # GIRVAN-NEWMAN
    # ==========================================

    print("\n----- Girvan-Newman -----")

    (
        gn_partition,
        gn_modularity
    ) = detect_girvan_newman(
        G_comparison,
        max_levels=30
    )

    gn_communities = len(
        gn_partition
    )

    print(
        "\nBest GN result within "
        "evaluated levels:"
    )

    print(
        "Communities:",
        gn_communities
    )

    print(
        "Modularity:",
        gn_modularity
    )

    # ==========================================
    # SAVE COMPARISON TABLE
    # ==========================================

    comparison = pd.DataFrame(
        [
            {
                "Method": "Louvain",
                "Nodes": (
                    G_comparison.number_of_nodes()
                ),
                "Edges": (
                    G_comparison.number_of_edges()
                ),
                "Communities": (
                    louvain_communities
                ),
                "Modularity": (
                    louvain_modularity
                )
            },
            {
                "Method": "Girvan-Newman",
                "Nodes": (
                    G_comparison.number_of_nodes()
                ),
                "Edges": (
                    G_comparison.number_of_edges()
                ),
                "Communities": (
                    gn_communities
                ),
                "Modularity": (
                    gn_modularity
                )
            }
        ]
    )

    COMPARISON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False
    )

    print(
        f"\nSaved comparison to: "
        f"{COMPARISON_FILE}"
    )

    # ==========================================
    # PLOT MODULARITY
    # ==========================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        comparison["Method"],
        comparison["Modularity"]
    )

    plt.ylabel(
        "Modularity"
    )

    plt.xlabel(
        "Community Detection Method"
    )

    plt.title(
        "Louvain vs Girvan-Newman "
        "Community Detection"
    )

    plt.tight_layout()

    COMPARISON_FIGURE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        COMPARISON_FIGURE,
        dpi=300
    )

    plt.close()

    print(
        f"Saved comparison figure to: "
        f"{COMPARISON_FIGURE}"
    )


def main():

    # ==========================================
    # LOAD DATA
    # ==========================================

    df = pd.read_csv(
        PROCESSED_FILE
    )

    # ==========================================
    # BUILD NETWORK
    # ==========================================

    G = build_graph(df)

    print(
        "Nodes:",
        G.number_of_nodes()
    )

    print(
        "Edges:",
        G.number_of_edges()
    )

    # ==========================================
    # FULL-NETWORK LOUVAIN
    # ==========================================

    print(
        "\n===== FULL NETWORK LOUVAIN ====="
    )

    partition = detect_louvain(
        G
    )

    number_of_communities = len(
        set(
            partition.values()
        )
    )

    modularity = (
        community_louvain.modularity(
            partition,
            G,
            weight="weight"
        )
    )

    print(
        "Number of communities:",
        number_of_communities
    )

    print(
        "Modularity:",
        modularity
    )

    # ==========================================
    # COMMUNITY SIZES
    # ==========================================

    community_sizes = {}

    for community_id in (
        partition.values()
    ):

        community_sizes[
            community_id
        ] = (
            community_sizes.get(
                community_id,
                0
            ) + 1
        )

    print(
        "\nCommunity sizes:"
    )

    for (
        community_id,
        size
    ) in sorted(
        community_sizes.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"Community {community_id}: "
            f"{size} nodes"
        )

    # ==========================================
    # SAVE LOUVAIN ASSIGNMENTS
    # ==========================================

    result = pd.DataFrame(
        partition.items(),
        columns=[
            "user",
            "community"
        ]
    )

    result = result.sort_values(
        "user"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved community assignments to: "
        f"{OUTPUT_FILE}"
    )

    # ==========================================
    # CONTROLLED COMMUNITY COMPARISON
    # ==========================================

    run_community_comparison(
        G
    )


if __name__ == "__main__":
    main()
