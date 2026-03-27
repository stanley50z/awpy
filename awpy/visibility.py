"""Module for calculating visibility.

Reference: https://github.com/AtomicBool/cs2-map-parser
"""

from __future__ import annotations

import pathlib
import struct
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from loguru import logger
import numba

import awpy.vector


@dataclass
class Triangle:
    """A triangle in 3D space defined by three vertices.

    Attributes:
        p1: First vertex of the triangle.
        p2: Second vertex of the triangle.
        p3: Third vertex of the triangle.
    """

    p1: awpy.vector.Vector3
    p2: awpy.vector.Vector3
    p3: awpy.vector.Vector3

    def get_centroid(self) -> awpy.vector.Vector3:
        """Calculate the centroid of the triangle.

        Returns:
            awpy.vector.Vector3: Centroid of the triangle.
        """
        return awpy.vector.Vector3(
            (self.p1.x + self.p2.x + self.p3.x) / 3,
            (self.p1.y + self.p2.y + self.p3.y) / 3,
            (self.p1.z + self.p2.z + self.p3.z) / 3,
        )


@dataclass
class Edge:
    """An edge in a triangulated mesh.

    Attributes:
        next: Index of the next edge in the face.
        twin: Index of the twin edge in the adjacent face.
        origin: Index of the vertex where this edge starts.
        face: Index of the face this edge belongs to.
    """

    next: int
    twin: int
    origin: int
    face: int


class KV3Parser:
    """Parser for KV3 format files used in Source 2 engine.

    This class provides functionality to parse KV3 files, which are used to store
    various game data including physics collision meshes.

    Attributes:
        content: Raw content of the KV3 file.
        index: Current parsing position in the content.
        parsed_data: Resulting parsed data structure.
    """

    def __init__(self) -> None:
        """Initialize a new KV3Parser instance."""
        self.content = ""
        self.index = 0
        self.parsed_data = None

    def parse(self, content: str) -> None:
        """Parse the given KV3 content string.

        Args:
            content: String containing KV3 formatted data.
        """
        self.content = content
        self.index = 0
        self._skip_until_first_bracket()
        self.parsed_data = self._parse_value()

    def get_value(self, path: str) -> str:
        """Get a value from the parsed data using a dot-separated path.

        Args:
            path: Dot-separated path to the desired value, e.g.,
                "section.subsection[0].value"

        Returns:
            String value at the specified path, or empty string
                if not found.
        """
        if not self.parsed_data:
            return ""

        current = self.parsed_data
        for segment in path.split("."):
            key = segment
            array_index = None

            if "[" in segment:
                key = segment[: segment.find("[")]
                array_index = int(segment[segment.find("[") + 1 : segment.find("]")])

            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return ""

            if array_index is not None:
                if isinstance(current, list) and array_index < len(current):
                    current = current[array_index]
                else:
                    return ""

        return current if isinstance(current, str) else ""

    def _skip_until_first_bracket(self) -> None:
        """Skip content until the first opening bracket is found."""
        while self.index < len(self.content) and self.content[self.index] != "{":
            self.index = self.content.find("\n", self.index) + 1

    def _skip_whitespace(self) -> None:
        """Skip all whitespace characters at the current position."""
        while self.index < len(self.content) and self.content[self.index].isspace():
            self.index += 1

    def _parse_value(self) -> dict | list | str | None:
        """Parse a value from the current position.

        Returns:
            Parsed value which can be a dictionary, list, or string,
                or None if parsing fails.
        """
        self._skip_whitespace()
        if self.index >= len(self.content):
            return None

        char = self.content[self.index]
        if char == "{":
            return self._parse_object()
        if char == "[":
            return self._parse_array()
        if char == "#" and self.index + 1 < len(self.content) and self.content[self.index + 1] == "[":
            self.index += 1
            return self._parse_byte_array()
        return self._parse_string()

    def _parse_object(self) -> dict:
        """Parse a KV3 object starting at the current position.

        Returns:
            Dictionary containing the parsed key-value pairs.
        """
        self.index += 1  # Skip {
        obj = {}
        while self.index < len(self.content):
            self._skip_whitespace()
            if self.content[self.index] == "}":
                self.index += 1
                return obj

            key = self._parse_string()
            self._skip_whitespace()
            if self.content[self.index] == "=":
                self.index += 1

            value = self._parse_value()
            if key and value is not None:
                obj[key] = value

            self._skip_whitespace()
            if self.content[self.index] == ",":
                self.index += 1

        return obj

    def _parse_array(self) -> list:
        """Parse a KV3 array starting at the current position.

        Returns:
            List containing the parsed values.
        """
        self.index += 1  # Skip [
        arr = []
        while self.index < len(self.content):
            self._skip_whitespace()
            if self.content[self.index] == "]":
                self.index += 1
                return arr

            value = self._parse_value()
            if value is not None:
                arr.append(value)

            self._skip_whitespace()
            if self.content[self.index] == ",":
                self.index += 1
        return arr

    def _parse_byte_array(self) -> str:
        """Parse a KV3 byte array starting at the current position.

        Returns:
            Space-separated string of byte values.
        """
        self.index += 1  # Skip [
        start = self.index
        while self.index < len(self.content) and self.content[self.index] != "]":
            self.index += 1
        byte_str = self.content[start : self.index].strip()
        self.index += 1  # Skip ]
        return " ".join(byte_str.split())

    def _parse_string(self) -> str:
        """Parse a string value at the current position.

        Returns:
            Parsed string value.
        """
        start = self.index
        while self.index < len(self.content):
            char = self.content[self.index]
            if char in "={}[], \n":
                break
            self.index += 1
        return self.content[start : self.index].strip()


class VphysParser:
    """Parser for VPhys collision files.

    This class extracts and processes collision geometry data
        from VPhys files, converting it into a set of triangles.

    Attributes:
        vphys_file (Path): Path to the VPhys file.
        triangles (list[Triangle]): List of parsed triangles from the VPhys file.
        kv3_parser (KV3Parser): Helper parser for extracting key-value data from
            the .vphys file.
    """

    def __init__(self, vphys_file: str | pathlib.Path) -> None:
        """Initializes the parser with the path to a VPhys file.

        Args:
            vphys_file (str | pathlib.Path): Path to the VPhys file
                to parse.
        """
        self.vphys_file = pathlib.Path(vphys_file)
        self.triangles: list[Triangle] = []
        self.kv3_parser = KV3Parser()
        self.parse()

    @overload
    @staticmethod
    def bytes_to_vec(byte_str: str, element_type: Literal["uint8", "int32"]) -> list[int]: ...

    @overload
    @staticmethod
    def bytes_to_vec(byte_str: str, element_type: Literal["float"]) -> list[float]: ...

    @staticmethod
    def bytes_to_vec(byte_str: str, element_type: Literal["uint8", "int32", "float"]) -> list[int] | list[float]:
        """Converts a space-separated string of byte values into a list of numbers.

        Args:
            byte_str (str): Space-separated string of hexadecimal byte values.
            element_type (int): Types represented by the bytes (uint8, int32, float).

        Returns:
            list[int | float]: List of converted values (integers for
                uint8, floats for size 4).
        """
        bytes_list = [int(b, 16) for b in byte_str.split()]
        result = []

        if element_type == "uint8":
            return bytes_list

        element_size = 4  # For int and float

        # Convert bytes to appropriate type based on size
        for i in range(0, len(bytes_list), element_size):
            chunk = bytes(bytes_list[i : i + element_size])
            if element_type == "float":  # float
                val = struct.unpack("f", chunk)[0]
                result.append(val)
            else:  # int32
                val = struct.unpack("i", chunk)[0]
                result.append(val)
        return result

    def get_collision_attribute_indices_for_default_group(self) -> list[str]:
        """Get collision attribute indices for the default group.

        Returns:
            list[int]: List of collision attribute indices for the default group.
        """
        collision_attribute_indices = []
        idx = 0
        while True:
            collision_group_string = self.kv3_parser.get_value(f"m_collisionAttributes[{idx}].m_CollisionGroupString")
            if not collision_group_string:
                break
            if collision_group_string.lower() == '"default"':
                collision_attribute_indices.append(str(idx))
            idx += 1
        return collision_attribute_indices

    def parse(self) -> None:
        """Parses the VPhys file and extracts collision geometry.

        Processes hulls and meshes in the VPhys file to generate a list of triangles.
        """
        if len(self.triangles) > 0:
            logger.debug(f"VPhys data already parsed, got {len(self.triangles)} triangles.")
            return

        logger.debug(f"Parsing vphys file: {self.vphys_file}")

        # Read file
        with open(self.vphys_file) as f:
            data = f.read()

        # Parse VPhys data
        self.kv3_parser.parse(data)

        collision_attribute_indices = self.get_collision_attribute_indices_for_default_group()

        logger.debug(f"Extracted collision attribute indices: {collision_attribute_indices}")

        # Process hulls
        hull_idx = 0
        hull_count = 0
        while True:
            if hull_idx % 1000 == 0:
                logger.debug(f"Processing hull {hull_idx}...")

            collision_idx = self.kv3_parser.get_value(
                f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_nCollisionAttributeIndex"
            )
            if not collision_idx:
                break

            if collision_idx in collision_attribute_indices:
                # Get vertices
                vertex_str = self.kv3_parser.get_value(
                    f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_Hull.m_VertexPositions"
                )
                if not vertex_str:
                    vertex_str = self.kv3_parser.get_value(
                        f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_Hull.m_Vertices"
                    )

                vertex_data = self.bytes_to_vec(vertex_str, "float")
                vertices = [
                    awpy.vector.Vector3(vertex_data[i], vertex_data[i + 1], vertex_data[i + 2])
                    for i in range(0, len(vertex_data), 3)
                ]

                # Get faces and edges
                faces = self.bytes_to_vec(
                    self.kv3_parser.get_value(f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_Hull.m_Faces"),
                    "uint8",
                )
                edge_data = self.bytes_to_vec(
                    self.kv3_parser.get_value(f"m_parts[0].m_rnShape.m_hulls[{hull_idx}].m_Hull.m_Edges"),
                    "uint8",
                )

                edges = [
                    Edge(
                        edge_data[i],
                        edge_data[i + 1],
                        edge_data[i + 2],
                        edge_data[i + 3],
                    )
                    for i in range(0, len(edge_data), 4)
                ]

                # Process triangles
                for start_edge in faces:
                    edge = edges[start_edge].next
                    while edge != start_edge:
                        next_edge = edges[edge].next
                        self.triangles.append(
                            Triangle(
                                vertices[edges[start_edge].origin],
                                vertices[edges[edge].origin],
                                vertices[edges[next_edge].origin],
                            )
                        )
                        edge = next_edge

                hull_count += 1
            hull_idx += 1

        # Process meshes
        mesh_idx = 0
        mesh_count = 0
        while True:
            logger.debug(f"Processing mesh {mesh_idx}...")
            collision_idx = self.kv3_parser.get_value(
                f"m_parts[0].m_rnShape.m_meshes[{mesh_idx}].m_nCollisionAttributeIndex"
            )
            if not collision_idx:
                break

            if collision_idx in collision_attribute_indices:
                # Get triangles and vertices
                tri_data = self.bytes_to_vec(
                    self.kv3_parser.get_value(f"m_parts[0].m_rnShape.m_meshes[{mesh_idx}].m_Mesh.m_Triangles"),
                    "int32",
                )
                vertex_data = self.bytes_to_vec(
                    self.kv3_parser.get_value(f"m_parts[0].m_rnShape.m_meshes[{mesh_idx}].m_Mesh.m_Vertices"),
                    "float",
                )

                vertices = [
                    awpy.vector.Vector3(vertex_data[i], vertex_data[i + 1], vertex_data[i + 2])
                    for i in range(0, len(vertex_data), 3)
                ]

                for i in range(0, len(tri_data), 3):
                    self.triangles.append(
                        Triangle(
                            vertices[int(tri_data[i])],
                            vertices[int(tri_data[i + 1])],
                            vertices[int(tri_data[i + 2])],
                        )
                    )

                mesh_count += 1
            mesh_idx += 1

    def to_tri(self, path: str | pathlib.Path | None) -> None:
        """Export parsed triangles to a .tri file.

        Args:
            path: Path to the output .tri file.
        """
        if not path:
            path = self.vphys_file.with_suffix(".tri")
        outpath = pathlib.Path(path)

        logger.debug(f"Exporting {len(self.triangles)} triangles to {outpath}")
        with open(outpath, "wb") as f:
            for triangle in self.triangles:
                # Write all awpy.vector.Vector3 components as float32
                f.write(struct.pack("f", triangle.p1.x))
                f.write(struct.pack("f", triangle.p1.y))
                f.write(struct.pack("f", triangle.p1.z))
                f.write(struct.pack("f", triangle.p2.x))
                f.write(struct.pack("f", triangle.p2.y))
                f.write(struct.pack("f", triangle.p2.z))
                f.write(struct.pack("f", triangle.p3.x))
                f.write(struct.pack("f", triangle.p3.y))
                f.write(struct.pack("f", triangle.p3.z))

        logger.success(f"Processed {len(self.triangles)} triangles from {self.vphys_file} -> {outpath}")


class AABB:
    """Axis-Aligned Bounding Box for efficient collision detection."""

    def __init__(self, min_point: awpy.vector.Vector3, max_point: awpy.vector.Vector3) -> None:
        """Initialize the AABB with minimum and maximum points.

        Args:
            min_point (awpy.vector.Vector3): Minimum point of the AABB.
            max_point (awpy.vector.Vector3): Maximum point of the AABB.
        """
        self.min_point = min_point
        self.max_point = max_point

    @classmethod
    def from_triangle(cls, triangle: Triangle) -> AABB:
        """Create an AABB from a triangle.

        Args:
            triangle (Triangle): Triangle to create the AABB from.

        Returns:
            AABB: Axis-Aligned Bounding Box encompassing the triangle.
        """
        min_point = awpy.vector.Vector3(
            min(triangle.p1.x, triangle.p2.x, triangle.p3.x),
            min(triangle.p1.y, triangle.p2.y, triangle.p3.y),
            min(triangle.p1.z, triangle.p2.z, triangle.p3.z),
        )
        max_point = awpy.vector.Vector3(
            max(triangle.p1.x, triangle.p2.x, triangle.p3.x),
            max(triangle.p1.y, triangle.p2.y, triangle.p3.y),
            max(triangle.p1.z, triangle.p2.z, triangle.p3.z),
        )
        return cls(min_point, max_point)

    def intersects_ray(self, ray_origin: awpy.vector.Vector3, ray_direction: awpy.vector.Vector3) -> bool:
        """Check if a ray intersects with the AABB.

        Args:
            ray_origin (awpy.vector.Vector3): Ray origin point.
            ray_direction (awpy.vector.Vector3): Ray direction vector.

        Returns:
            bool: True if the ray intersects with the AABB, False otherwise.
        """
        epsilon = 1e-6

        def check_axis(origin: float, direction: float, min_val: float, max_val: float) -> tuple[float, float]:
            if abs(direction) < epsilon:
                if origin < min_val or origin > max_val:
                    return float("inf"), float("-inf")
                return float("-inf"), float("inf")

            t1 = (min_val - origin) / direction
            t2 = (max_val - origin) / direction
            return (min(t1, t2), max(t1, t2))

        tx_min, tx_max = check_axis(ray_origin.x, ray_direction.x, self.min_point.x, self.max_point.x)
        ty_min, ty_max = check_axis(ray_origin.y, ray_direction.y, self.min_point.y, self.max_point.y)
        tz_min, tz_max = check_axis(ray_origin.z, ray_direction.z, self.min_point.z, self.max_point.z)

        t_enter = max(tx_min, ty_min, tz_min)
        t_exit = min(tx_max, ty_max, tz_max)

        return t_enter <= t_exit and t_exit >= 0


class BVHNode:
    """Node in the Bounding Volume Hierarchy tree."""

    def __init__(
        self,
        aabb: AABB,
        triangle: Triangle | None = None,
        left: BVHNode | None = None,
        right: BVHNode | None = None,
    ) -> None:
        """Initialize a BVHNode with an AABB and optional triangle and children.

        Args:
            aabb (AABB): Axis-Aligned Bounding Box of the node.
            triangle (Triangle | None, optional): Triangle contained
                in the node. Defaults to None.
            left (BVHNode | None, optional): Left child node. Defaults to None.
            right (BVHNode | None, optional): Right child node. Defaults to None.
        """
        self.aabb = aabb
        self.triangle = triangle
        self.left = left
        self.right = right


def read_tri_flat(tri_file: str | pathlib.Path) -> np.ndarray:
    """Read a .tri file into a flat (N, 9) float64 numpy array.

    Each row is [p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z].
    """
    tri_file = pathlib.Path(tri_file)
    raw = np.fromfile(tri_file, dtype=np.float32)
    n_triangles = len(raw) // 9
    return raw[: n_triangles * 9].reshape(n_triangles, 9).astype(np.float64)


def build_flat_bvh(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a flattened BVH from an (N, 9) triangle array.

    Returns:
        nodes: float64 (M, 6) - AABB [min_x, min_y, min_z, max_x, max_y, max_z]
        children: int32 (M, 2) - [left, right] child indices (-1 for leaves)
        tri_idx: int32 (M,) - triangle index for leaves (-1 for internal)
    """
    n = len(triangles)

    centroids = (
        triangles[:, 0:3] + triangles[:, 3:6] + triangles[:, 6:9]
    ) / 3.0

    p1 = triangles[:, 0:3]
    p2 = triangles[:, 3:6]
    p3 = triangles[:, 6:9]
    tri_mins = np.minimum(np.minimum(p1, p2), p3)
    tri_maxs = np.maximum(np.maximum(p1, p2), p3)

    max_nodes = 2 * n - 1
    nodes = np.empty((max_nodes, 6), dtype=np.float64)
    children = np.full((max_nodes, 2), -1, dtype=np.int32)
    tri_idx = np.full(max_nodes, -1, dtype=np.int32)
    node_count = 0

    indices = np.arange(n, dtype=np.int32)

    root_idx = 0
    node_count = 1

    stack: list[tuple[int, np.ndarray]] = [(root_idx, indices)]

    while stack:
        nid, idx = stack.pop()

        if len(idx) == 1:
            i = idx[0]
            nodes[nid, :3] = tri_mins[i]
            nodes[nid, 3:] = tri_maxs[i]
            tri_idx[nid] = i
            continue

        subset_mins = tri_mins[idx]
        subset_maxs = tri_maxs[idx]
        aabb_min = subset_mins.min(axis=0)
        aabb_max = subset_maxs.max(axis=0)
        nodes[nid, :3] = aabb_min
        nodes[nid, 3:] = aabb_max

        c = centroids[idx]
        spreads = c.max(axis=0) - c.min(axis=0)
        axis = int(np.argmax(spreads))

        mid = len(idx) // 2
        order = np.argpartition(c[:, axis], mid)
        left_idx = idx[order[:mid]]
        right_idx = idx[order[mid:]]

        left_nid = node_count
        right_nid = node_count + 1
        node_count += 2

        children[nid, 0] = left_nid
        children[nid, 1] = right_nid

        stack.append((left_nid, left_idx))
        stack.append((right_nid, right_idx))

    nodes = nodes[:node_count]
    children = children[:node_count]
    tri_idx = tri_idx[:node_count]

    return nodes, children, tri_idx


@numba.njit(cache=True)
def _aabb_intersects_ray(
    box_min_x: float, box_min_y: float, box_min_z: float,
    box_max_x: float, box_max_y: float, box_max_z: float,
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
) -> bool:
    """Slab-method AABB-ray intersection test."""
    eps = 1e-6
    inv_dx = 1.0 / dx if abs(dx) > eps else 1e18 if dx >= 0 else -1e18
    inv_dy = 1.0 / dy if abs(dy) > eps else 1e18 if dy >= 0 else -1e18
    inv_dz = 1.0 / dz if abs(dz) > eps else 1e18 if dz >= 0 else -1e18

    tx1 = (box_min_x - ox) * inv_dx
    tx2 = (box_max_x - ox) * inv_dx
    if tx1 > tx2:
        tx1, tx2 = tx2, tx1

    ty1 = (box_min_y - oy) * inv_dy
    ty2 = (box_max_y - oy) * inv_dy
    if ty1 > ty2:
        ty1, ty2 = ty2, ty1

    tz1 = (box_min_z - oz) * inv_dz
    tz2 = (box_max_z - oz) * inv_dz
    if tz1 > tz2:
        tz1, tz2 = tz2, tz1

    t_enter = max(tx1, ty1, tz1)
    t_exit = min(tx2, ty2, tz2)
    return t_enter <= t_exit and t_exit >= 0.0


@numba.njit(cache=True)
def _ray_tri_intersect(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    p1x: float, p1y: float, p1z: float,
    p2x: float, p2y: float, p2z: float,
    p3x: float, p3y: float, p3z: float,
    max_dist: float,
) -> bool:
    """Moller-Trumbore ray-triangle intersection. Returns True if hit within max_dist."""
    eps = 1e-6
    e1x = p2x - p1x; e1y = p2y - p1y; e1z = p2z - p1z
    e2x = p3x - p1x; e2y = p3y - p1y; e2z = p3z - p1z

    hx = dy * e2z - dz * e2y
    hy = dz * e2x - dx * e2z
    hz = dx * e2y - dy * e2x
    a = e1x * hx + e1y * hy + e1z * hz
    if -eps < a < eps:
        return False

    f = 1.0 / a
    sx = ox - p1x; sy = oy - p1y; sz = oz - p1z
    u = f * (sx * hx + sy * hy + sz * hz)
    if u < 0.0 or u > 1.0:
        return False

    qx = sy * e1z - sz * e1y
    qy = sz * e1x - sx * e1z
    qz = sx * e1y - sy * e1x
    v = f * (dx * qx + dy * qy + dz * qz)
    if v < 0.0 or u + v > 1.0:
        return False

    t = f * (e2x * qx + e2y * qy + e2z * qz)
    return t > eps and t <= max_dist


@numba.njit(cache=True)
def _traverse_flat_bvh(
    nodes: np.ndarray,       # (M, 6) float64
    children: np.ndarray,    # (M, 2) int32
    tri_idx: np.ndarray,    # (M,) int32
    triangles: np.ndarray,  # (N, 9) float64
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    max_dist: float,
) -> bool:
    """Iterative BVH traversal using explicit stack. Returns True if any hit."""
    stack = np.empty(64, dtype=np.int32)  # 64 levels is enough for billions of triangles
    stack_top = 0
    stack[0] = 0  # root node
    stack_top = 1

    while stack_top > 0:
        stack_top -= 1
        nid = stack[stack_top]

        n = nodes[nid]
        if not _aabb_intersects_ray(n[0], n[1], n[2], n[3], n[4], n[5],
                                     ox, oy, oz, dx, dy, dz):
            continue

        t = tri_idx[nid]
        if t >= 0:
            # Leaf node
            tri = triangles[t]
            if _ray_tri_intersect(ox, oy, oz, dx, dy, dz,
                                  tri[0], tri[1], tri[2],
                                  tri[3], tri[4], tri[5],
                                  tri[6], tri[7], tri[8],
                                  max_dist):
                return True
        else:
            # Internal node - push children
            left = children[nid, 0]
            right = children[nid, 1]
            if left >= 0:
                stack[stack_top] = left
                stack_top += 1
            if right >= 0:
                stack[stack_top] = right
                stack_top += 1

    return False


def is_visible_flat(
    nodes: np.ndarray,
    children: np.ndarray,
    tri_idx: np.ndarray,
    triangles: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> bool:
    """Check visibility using the flat BVH. Pure-numpy/numba path.

    Args:
        nodes, children, tri_idx: from build_flat_bvh()
        triangles: (N, 9) float64 array from read_tri_flat()
        start: (3,) float64 array
        end: (3,) float64 array

    Returns:
        True if line of sight is clear, False if blocked.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
    if dist < 1e-6:
        return True
    dx /= dist
    dy /= dist
    dz /= dist
    return not _traverse_flat_bvh(
        nodes, children, tri_idx, triangles,
        start[0], start[1], start[2],
        dx, dy, dz, dist,
    )


_BVH_MAGIC = b"AWBVH001"  # 8-byte magic for format versioning


def save_flat_bvh(
    path: str | pathlib.Path,
    nodes: np.ndarray,
    children: np.ndarray,
    tri_idx: np.ndarray,
) -> None:
    """Save a flat BVH to a binary file.

    Format: magic (8B) | node_count (4B) | nodes (M*48B) | children (M*8B) | tri_idx (M*4B)
    """
    path = pathlib.Path(path)
    m = len(nodes)
    with open(path, "wb") as f:
        f.write(_BVH_MAGIC)
        f.write(struct.pack("<I", m))
        f.write(nodes.astype(np.float64).tobytes())
        f.write(children.astype(np.int32).tobytes())
        f.write(tri_idx.astype(np.int32).tobytes())


def load_flat_bvh(path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a flat BVH from a binary file.

    Returns:
        nodes: (M, 6) float64
        children: (M, 2) int32
        tri_idx: (M,) int32
    """
    path = pathlib.Path(path)
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _BVH_MAGIC:
            raise ValueError(f"Invalid BVH file: bad magic {magic!r}")
        m = struct.unpack("<I", f.read(4))[0]
        nodes = np.frombuffer(f.read(m * 6 * 8), dtype=np.float64).reshape(m, 6).copy()
        children = np.frombuffer(f.read(m * 2 * 4), dtype=np.int32).reshape(m, 2).copy()
        tri_idx = np.frombuffer(f.read(m * 4), dtype=np.int32).copy()
    return nodes, children, tri_idx


@numba.njit(cache=True, parallel=True)
def _is_visible_batch_kernel(
    nodes: np.ndarray,
    children: np.ndarray,
    tri_idx_arr: np.ndarray,
    triangles: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    results: np.ndarray,
) -> None:
    """Check visibility for a batch of ray pairs in parallel."""
    n = len(starts)
    for i in numba.prange(n):
        sx, sy, sz = starts[i, 0], starts[i, 1], starts[i, 2]
        ex, ey, ez = ends[i, 0], ends[i, 1], ends[i, 2]
        dx = ex - sx
        dy = ey - sy
        dz = ez - sz
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist < 1e-6:
            results[i] = True
            continue
        dx /= dist
        dy /= dist
        dz /= dist
        results[i] = not _traverse_flat_bvh(
            nodes, children, tri_idx_arr, triangles,
            sx, sy, sz, dx, dy, dz, dist,
        )


class VisibilityChecker:
    """Class for visibility checking in 3D space using a BVH structure."""

    def __init__(self, path: pathlib.Path | None = None, triangles: list[Triangle] | None = None) -> None:
        """Initialize the visibility checker with a list of triangles.

        Args:
            path (pathlib.Path | None, optional): Path to a .tri file to read
                triangles from.
            triangles (list[Triangle] | None, optional): List of triangles to
                build the BVH from.
        """
        self._path = path
        self._triangles_list = triangles
        self._root: BVHNode | None = None

        if path is not None:
            self._tri_flat = read_tri_flat(path)
            self.n_triangles = len(self._tri_flat)

            bvh_path = pathlib.Path(path).with_suffix(".bvh")
            tri_mtime = pathlib.Path(path).stat().st_mtime

            if bvh_path.exists() and bvh_path.stat().st_mtime >= tri_mtime:
                logger.debug(f"Loading cached BVH from {bvh_path}")
                self._bvh_nodes, self._bvh_children, self._bvh_tri_idx = load_flat_bvh(bvh_path)
            else:
                logger.debug(f"Building BVH for {path} ({self.n_triangles} triangles)")
                self._bvh_nodes, self._bvh_children, self._bvh_tri_idx = build_flat_bvh(self._tri_flat)
                try:
                    save_flat_bvh(bvh_path, self._bvh_nodes, self._bvh_children, self._bvh_tri_idx)
                    logger.debug(f"Cached BVH to {bvh_path}")
                except OSError:
                    logger.warning(f"Could not cache BVH to {bvh_path}")

        elif triangles is not None:
            self._tri_flat = np.array(
                [[t.p1.x, t.p1.y, t.p1.z, t.p2.x, t.p2.y, t.p2.z, t.p3.x, t.p3.y, t.p3.z] for t in triangles],
                dtype=np.float64,
            )
            self.n_triangles = len(self._tri_flat)
            self._bvh_nodes, self._bvh_children, self._bvh_tri_idx = build_flat_bvh(self._tri_flat)
        else:
            raise ValueError("Either path or triangles must be provided")

    @property
    def root(self) -> BVHNode:
        """Legacy BVH tree root. Built on first access."""
        if self._root is None:
            if self._path is not None:
                triangles = self.read_tri_file(self._path)
            elif self._triangles_list is not None:
                triangles = self._triangles_list
            else:
                raise RuntimeError("Cannot build legacy BVH: no source data")
            self._root = self._build_bvh(triangles)
        return self._root

    @root.setter
    def root(self, value: BVHNode | None) -> None:
        self._root = value

    def __repr__(self) -> str:
        """Return a string representation of the VisibilityChecker."""
        return f"VisibilityChecker(n_triangles={self.n_triangles})"

    def _build_bvh(self, triangles: list[Triangle]) -> BVHNode:
        """Build a BVH tree from a list of triangles.

        Args:
            triangles (list[Triangle]): List of triangles to build the BVH from.

        Returns:
            BVHNode: Root node of the BVH tree.
        """
        if len(triangles) == 1:
            return BVHNode(AABB.from_triangle(triangles[0]), triangle=triangles[0])

        # Calculate centroids and find split axis
        centroids = [t.get_centroid() for t in triangles]

        # Find the axis with the largest spread
        min_x = min(c.x for c in centroids)
        max_x = max(c.x for c in centroids)
        min_y = min(c.y for c in centroids)
        max_y = max(c.y for c in centroids)
        min_z = min(c.z for c in centroids)
        max_z = max(c.z for c in centroids)

        x_spread = max_x - min_x
        y_spread = max_y - min_y
        z_spread = max_z - min_z

        # Choose split axis
        if x_spread >= y_spread and x_spread >= z_spread:
            axis = 0  # x-axis
        elif y_spread >= z_spread:
            axis = 1  # y-axis
        else:
            axis = 2  # z-axis

        # Sort triangles based on centroid position
        triangles = sorted(
            triangles,
            key=lambda t: (
                t.get_centroid().x if axis == 0 else t.get_centroid().y if axis == 1 else t.get_centroid().z
            ),
        )

        # Split triangles into two groups
        mid = len(triangles) // 2
        left = self._build_bvh(triangles[:mid])
        right = self._build_bvh(triangles[mid:])

        # Create encompassing AABB
        min_point = awpy.vector.Vector3(
            min(left.aabb.min_point.x, right.aabb.min_point.x),
            min(left.aabb.min_point.y, right.aabb.min_point.y),
            min(left.aabb.min_point.z, right.aabb.min_point.z),
        )
        max_point = awpy.vector.Vector3(
            max(left.aabb.max_point.x, right.aabb.max_point.x),
            max(left.aabb.max_point.y, right.aabb.max_point.y),
            max(left.aabb.max_point.z, right.aabb.max_point.z),
        )

        return BVHNode(AABB(min_point, max_point), left=left, right=right)

    def _ray_triangle_intersection(
        self,
        ray_origin: awpy.vector.Vector3,
        ray_direction: awpy.vector.Vector3,
        triangle: Triangle,
    ) -> float | None:
        """Check if a ray intersects with a triangle.

        Args:
            ray_origin (awpy.vector.Vector3): Ray origin point.
            ray_direction (awpy.vector.Vector3): Ray direction vector.
            triangle (Triangle): Triangle to check intersection with.

        Returns:
            float | None: Distance to the intersection point, or
                None if no intersection.
        """
        epsilon = 1e-6

        edge1 = triangle.p2 - triangle.p1
        edge2 = triangle.p3 - triangle.p1
        h = ray_direction.cross(edge2)
        a = edge1.dot(h)

        if -epsilon < a < epsilon:
            return None

        f = 1.0 / a
        s = ray_origin - triangle.p1
        u = f * s.dot(h)

        if u < 0.0 or u > 1.0:
            return None

        q = s.cross(edge1)
        v = f * ray_direction.dot(q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * edge2.dot(q)

        if t > epsilon:
            return t

        return None

    def _traverse_bvh(
        self,
        node: BVHNode,
        ray_origin: awpy.vector.Vector3,
        ray_direction: awpy.vector.Vector3,
        max_distance: float,
    ) -> bool:
        """Traverse the BVH tree to check for ray-triangle intersections.

        Args:
            node (BVHNode): Current node in the BVH tree.
            ray_origin (awpy.vector.Vector3): Ray origin point.
            ray_direction (awpy.vector.Vector3): Ray direction vector.
            max_distance (float): Maximum distance to check for intersections.

        Returns:
            bool: True if an intersection is found, False otherwise.
        """
        if not node.aabb.intersects_ray(ray_origin, ray_direction):
            return False

        # Leaf node - check triangle intersection
        if node.triangle:
            t = self._ray_triangle_intersection(ray_origin, ray_direction, node.triangle)
            return bool(t is not None and t <= max_distance)

        # Internal node - recurse through children
        return self._traverse_bvh(node.left, ray_origin, ray_direction, max_distance) or self._traverse_bvh(
            node.right, ray_origin, ray_direction, max_distance
        )

    def is_visible(
        self,
        start: awpy.vector.Vector3 | tuple | list,
        end: awpy.vector.Vector3 | tuple | list,
    ) -> bool:
        """Check if a line segment is visible in the 3D space.

        Args:
            start (awpy.vector.Vector3 | tuple | list): Start point of the line segment.
            end (awpy.vector.Vector3 | tuple | list): End point of the line segment.

        Returns:
            bool: True if the line segment is visible, False otherwise.
        """
        start_vec = awpy.vector.Vector3.from_input(start)
        end_vec = awpy.vector.Vector3.from_input(end)

        start_arr = np.array([start_vec.x, start_vec.y, start_vec.z], dtype=np.float64)
        end_arr = np.array([end_vec.x, end_vec.y, end_vec.z], dtype=np.float64)

        return is_visible_flat(
            self._bvh_nodes, self._bvh_children, self._bvh_tri_idx,
            self._tri_flat, start_arr, end_arr,
        )

    def is_visible_batch(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        """Check visibility for a batch of point pairs.

        Args:
            starts: (N, 3) float64 array of start points.
            ends: (N, 3) float64 array of end points.

        Returns:
            (N,) boolean array. True = visible, False = blocked.
        """
        starts = np.ascontiguousarray(starts, dtype=np.float64)
        ends = np.ascontiguousarray(ends, dtype=np.float64)
        results = np.empty(len(starts), dtype=np.bool_)
        _is_visible_batch_kernel(
            self._bvh_nodes, self._bvh_children, self._bvh_tri_idx,
            self._tri_flat, starts, ends, results,
        )
        return results

    @staticmethod
    def read_tri_file(tri_file: str | pathlib.Path, buffer_size: int = 1000) -> list[Triangle]:
        """Read triangles from a .tri file."""
        tri_file = pathlib.Path(tri_file)

        triangles: list[Triangle] = []

        with open(tri_file, "rb") as f:
            chunk_size = buffer_size * 9 * 4

            while True:
                data = f.read(chunk_size)
                if not data:
                    break

                num_floats = len(data) // 4
                num_complete_triangles = num_floats // 9

                for i in range(num_complete_triangles):
                    offset = i * 36
                    values = struct.unpack("9f", data[offset : offset + 36])

                    triangles.append(
                        Triangle(
                            awpy.vector.Vector3(values[0], values[1], values[2]),
                            awpy.vector.Vector3(values[3], values[4], values[5]),
                            awpy.vector.Vector3(values[6], values[7], values[8]),
                        )
                    )

        return triangles
