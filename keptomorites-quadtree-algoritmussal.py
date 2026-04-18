import math
import time
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx


# QUADTREE CSOMÓPONT
# Egy csomópont egy képrégiót reprezentál
# Színes kép esetén a value nem egyetlen szám,
# hanem egy RGB vektor: [R, G, B]
class QuadtreeNode:
    def __init__(self, x, y, width, height, value, variance, depth, is_leaf=True):
        # A blokk bal felső sarkának koordinátái
        self.x = x
        self.y = y

        # A blokk mérete
        self.width = width
        self.height = height

        # A blokk átlagos RGB színe
        self.value = value

        # A blokk varianciája (egyetlen szám a döntéshez)
        self.variance = variance

        # Mélység a fában
        self.depth = depth

        # Levél-e
        self.is_leaf = is_leaf

        # Gyerekek listája
        self.children = []


# SZÍNES KÉP BETÖLTÉSE
# A képet RGB módban tölti be
# A tömb alakja: (magasság, szélesség, 3)
def load_image_color(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    return arr


# RÉGIÓ STATISZTIKÁK
# Kiszámolja:
# - a blokk átlagos RGB színét
# - a blokk varianciáját csatornánként
# - a döntéshez egyetlen varianciaértéket használ
#   (az RGB varianciák átlaga)
def region_stats(image, x, y, width, height):
    region = image[y:y + height, x:x + width]   # shape: (h, w, 3)

    # Átlagos RGB szín
    mean_val = np.mean(region, axis=(0, 1))     # [R_avg, G_avg, B_avg]

    # Variancia csatornánként
    variance_rgb = np.var(region, axis=(0, 1))  # [var_R, var_G, var_B]

    # Egyetlen számot csinál belőle a döntéshez
    total_variance = float(np.mean(variance_rgb))

    return mean_val, total_variance


# QUADTREE ÉPÍTÉSE
def build_quadtree(image, x, y, width, height, threshold, max_depth, depth=0):
    mean_val, variance = region_stats(image, x, y, width, height)

    # Megállási feltételek:
    # - eléri a max mélységet
    # - 1 pixeles a blokk
    # - elég homogén a blokk
    if depth >= max_depth or width <= 1 or height <= 1 or variance <= threshold:
        return QuadtreeNode(
            x=x,
            y=y,
            width=width,
            height=height,
            value=mean_val,
            variance=variance,
            depth=depth,
            is_leaf=True
        )

    node = QuadtreeNode(
        x=x,
        y=y,
        width=width,
        height=height,
        value=mean_val,
        variance=variance,
        depth=depth,
        is_leaf=False
    )

    half_w = width // 2
    half_h = height // 2

    # Ha már nem lehet szabályosan 4 részre osztani
    if half_w == 0 or half_h == 0:
        node.is_leaf = True
        return node

    # 4 rész rekurzív feldolgozása
    top_left = build_quadtree(
        image, x, y, half_w, half_h, threshold, max_depth, depth + 1
    )

    top_right = build_quadtree(
        image, x + half_w, y, width - half_w, half_h, threshold, max_depth, depth + 1
    )

    bottom_left = build_quadtree(
        image, x, y + half_h, half_w, height - half_h, threshold, max_depth, depth + 1
    )

    bottom_right = build_quadtree(
        image, x + half_w, y + half_h, width - half_w, height - half_h, threshold, max_depth, depth + 1
    )

    node.children = [top_left, top_right, bottom_left, bottom_right]

    return node


# VÉGSŐ TÖMÖRÍTETT KÉP VISSZAÉPÍTÉSE
def reconstruct_image(node, output):
    if node.is_leaf:
        output[node.y:node.y + node.height, node.x:node.x + node.width] = node.value
    else:
        for child in node.children:
            reconstruct_image(child, output)


# REKONSTRUKCIÓ CSAK ADOTT MÉLYSÉGIG
# Fázisképekhez
# Ha elérjük a megadott mélységet, az adott blokkot
# az aktuális node átlagos színével töltjük ki.
def reconstruct_until_depth(node, output, max_draw_depth):
    if node.is_leaf or node.depth >= max_draw_depth:
        output[node.y:node.y + node.height, node.x:node.x + node.width] = node.value
    else:
        for child in node.children:
            reconstruct_until_depth(child, output, max_draw_depth)


# LEVÉLCSOMÓPONTOK SZÁMA
def count_leaves(node):
    if node.is_leaf:
        return 1
    return sum(count_leaves(child) for child in node.children)


# A FA MÉLYSÉGE
def tree_depth(node):
    if node.is_leaf:
        return node.depth
    return max(tree_depth(child) for child in node.children)


# MSE
# Színes képnél is működik: minden csatornát beleszámol
def mse(original, compressed):
    return float(np.mean((original - compressed) ** 2))


# PSNR
def psnr(original, compressed):
    error = mse(original, compressed)
    if error == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(error))


# LEVÉLBLOKKOK ÖSSZEGYŰJTÉSE
def collect_leaf_nodes(node, leaves=None):
    if leaves is None:
        leaves = []

    if node.is_leaf:
        leaves.append(node)
    else:
        for child in node.children:
            collect_leaf_nodes(child, leaves)

    return leaves


# ADOTT MÉLYSÉGIG AKTÍV BLOKKOK ÖSSZEGYŰJTÉSE
def collect_nodes_until_depth(node, max_draw_depth, nodes=None):
    if nodes is None:
        nodes = []

    if node.is_leaf or node.depth >= max_draw_depth:
        nodes.append(node)
    else:
        for child in node.children:
            collect_nodes_until_depth(child, max_draw_depth, nodes)

    return nodes


# TÖMÖRÍTÉSI ARÁNY BECSLÉSE
def estimated_compression_ratio(original_shape, leaf_count):
    height, width, _ = original_shape
    original_pixels = width * height

    # Egyszerű közelítés:
    # minden levélhez x, y, w, h + RGB
    # ezt leegyszerűsítve most 7 adattal becsüli
    quadtree_storage_estimate = leaf_count * 7

    if quadtree_storage_estimate == 0:
        return float("inf")

    return original_pixels / quadtree_storage_estimate


# EREDMÉNYEK MEGJELENÍTÉSE
def show_results(original, compressed, root, draw_boundaries=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(np.clip(original, 0, 255).astype(np.uint8))
    axes[0].set_title("Eredeti kép")
    axes[0].axis("off")

    axes[1].imshow(np.clip(compressed, 0, 255).astype(np.uint8))
    axes[1].set_title("Quadtree tömörített kép")
    axes[1].axis("off")

    if draw_boundaries:
        leaves = collect_leaf_nodes(root)
        for leaf in leaves:
            rect = Rectangle(
                (leaf.x, leaf.y),
                leaf.width,
                leaf.height,
                fill=False,
                linewidth=0.3,
                edgecolor="black"
            )
            axes[1].add_patch(rect)

    plt.tight_layout()
    plt.show()


# FÁZISKÉP MENTÉSE BLOKKHATÁROKKAL
def save_phase_image_with_boundaries(image_array, nodes, path, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(np.clip(image_array, 0, 255).astype(np.uint8))
    ax.axis("off")

    if title is not None:
        ax.set_title(title)

    for node in nodes:
        rect = Rectangle(
            (node.x, node.y),
            node.width,
            node.height,
            fill=False,
            linewidth=0.3,
            edgecolor="black"
        )
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close(fig)


# FÁZISKÉPEK MENTÉSE
def save_quadtree_phases(root, original_shape, max_phase_depth, output_dir="phases", draw_boundaries=True):
    os.makedirs(output_dir, exist_ok=True)

    for depth in range(max_phase_depth + 1):
        phase_image = np.zeros(original_shape, dtype=np.float32)

        # Rekonstrukció csak az adott mélységig
        reconstruct_until_depth(root, phase_image, depth)

        # Az adott mélységig aktív blokkok
        active_nodes = collect_nodes_until_depth(root, depth)

        filename = os.path.join(output_dir, f"phase_depth_{depth}.png")

        if draw_boundaries:
            save_phase_image_with_boundaries(
                phase_image,
                active_nodes,
                filename,
                title=f"Fáziskép - mélység {depth}"
            )
        else:
            save_image(phase_image, filename)

        #print(f"Mentve: {filename}") 


# KÉP MENTÉSE
def save_image(array, path):
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    img = Image.fromarray(clipped, mode="RGB")
    img.save(path)


# SZÖVEGES FA KIÍRÁS
# def print_tree(node, indent=0):
#     print(
#         " " * indent +
#         f"Node(depth={node.depth}, leaf={node.is_leaf}, "
#         f"x={node.x}, y={node.y}, w={node.width}, h={node.height}, "
#         f"var={node.variance:.2f})"
#     )

#     if not node.is_leaf:
#         for child in node.children:
#             print_tree(child, indent + 4)


# GRAFIKUS FA ELŐKÉSZÍTÉSE
def build_graph_from_quadtree(node, graph=None, parent_id=None, counter=None, max_draw_depth=None):
    if graph is None:
        graph = nx.DiGraph()

    if counter is None:
        counter = [0]

    node_id = counter[0]
    counter[0] += 1

    if max_draw_depth is not None and node.depth > max_draw_depth:
        return graph

    label = (
        f"d={node.depth}\n"
        f"{'leaf' if node.is_leaf else 'node'}\n"
        f"({node.x},{node.y})\n"
        f"{node.width}x{node.height}"
    )

    graph.add_node(node_id, label=label, depth=node.depth)

    if parent_id is not None:
        graph.add_edge(parent_id, node_id)

    if not node.is_leaf:
        for child in node.children:
            build_graph_from_quadtree(
                child,
                graph=graph,
                parent_id=node_id,
                counter=counter,
                max_draw_depth=max_draw_depth
            )

    return graph


# SZINTEKRE RENDEZETT POZÍCIÓK
def hierarchical_layout_by_levels(graph):
    levels = {}
    for node_id, data in graph.nodes(data=True):
        depth = data["depth"]
        if depth not in levels:
            levels[depth] = []
        levels[depth].append(node_id)

    pos = {}
    sorted_depths = sorted(levels.keys())

    for depth in sorted_depths:
        nodes_at_level = levels[depth]
        count = len(nodes_at_level)

        for i, node_id in enumerate(nodes_at_level):
            if count == 1:
                x = 0.5
            else:
                x = i / (count - 1)

            y = -depth
            pos[node_id] = (x, y)

    return pos


# QUADTREE FA KIRAJZOLÁSA
def draw_quadtree_tree(root, max_draw_depth=3):
    graph = build_graph_from_quadtree(root, max_draw_depth=max_draw_depth)
    pos = hierarchical_layout_by_levels(graph)
    labels = nx.get_node_attributes(graph, "label")

    plt.figure(figsize=(14, 8))
    nx.draw(
        graph,
        pos,
        labels=labels,
        with_labels=True,
        node_size=1800,
        font_size=7,
        arrows=False
    )
    plt.title("Quadtree fa struktúra (szintekre rendezve)")
    plt.axis("off")
    plt.show()


# FŐ PROGRAM
def main():
    # Bemeneti kép
    image_path = "zoldes.jpg"

    # Paraméterek
    threshold = 50 #darabos csokkent, reszletes novel
    #mikor tekintek egy blokkot homogénnek
    #ha a variancia ≤ threshold → nem bontjuk tovább
    #ha nagyobb → tovább osztjuk
    max_depth = 8

    # Hány mélységi szintig mentsünk fázisképeket
    phase_depth_to_save = max_depth

    # Kép betöltése
    original = load_image_color(image_path)

    # Quadtree építés időméréssel
    start_time = time.time()

    root = build_quadtree(
        image=original,
        x=0,
        y=0,
        width=original.shape[1],
        height=original.shape[0],
        threshold=threshold,
        max_depth=max_depth,
        depth=0
    )

    compressed = np.zeros_like(original)
    reconstruct_image(root, compressed)

    end_time = time.time()

    # Mérőszámok
    leaf_count = count_leaves(root)
    max_tree_depth = tree_depth(root)
    error_mse = mse(original, compressed)
    value_psnr = psnr(original, compressed)
    ratio_est = estimated_compression_ratio(original.shape, leaf_count)

    print("===== EREDMÉNYEK =====")
    print(f"Bemeneti kép mérete: {original.shape[1]} x {original.shape[0]} pixel")
    print(f"Threshold: {threshold}")
    print(f"Max depth: {max_depth}")
    print(f"Futási idő: {end_time - start_time:.4f} másodperc")
    print(f"Levélcsomópontok száma: {leaf_count}")
    print(f"A fa tényleges maximális mélysége: {max_tree_depth}")
    print(f"MSE: {error_mse:.4f}")
    print(f"PSNR: {value_psnr:.4f} dB")
    print(f"Becsült tömörítési arány: {ratio_est:.4f}")

    # Végső tömörített kép mentése
    save_image(compressed, "compressed_output_color.png")
    print("A tömörített színes kép elmentve: compressed_output_color.png")

    # Fázisképek mentése
    save_quadtree_phases(
        root=root,
        original_shape=original.shape,
        max_phase_depth=min(phase_depth_to_save, max_tree_depth),
        output_dir="phases_color",
        draw_boundaries=True
    )

    # Képek megjelenítése
    show_results(original, compressed, root, draw_boundaries=True)

    # Szöveges fa kiírás
    # print("\n===== QUADTREE SZÖVEGES KIÍRÁSA =====")
    # print_tree(root)

    # Quadtree fa kirajzolása
    draw_quadtree_tree(root, max_draw_depth=4)


if __name__ == "__main__":
    main()