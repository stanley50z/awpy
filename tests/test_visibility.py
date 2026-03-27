"""Test the visibility module."""

import numpy as np
import pytest
from click.testing import CliRunner

import awpy.cli
import awpy.data
import awpy.vector
import awpy.visibility


def check_visibility_brute_force(
    start: awpy.vector.Vector3 | tuple | list,
    end: awpy.vector.Vector3 | tuple | list,
    triangles: list[awpy.visibility.Triangle],
) -> bool:
    """Check visibility by testing against all triangles directly."""
    start_vec = awpy.vector.Vector3.from_input(start)
    end_vec = awpy.vector.Vector3.from_input(end)

    # Calculate ray direction and length
    direction = end_vec - start_vec
    distance = direction.length()

    if distance < 1e-6:
        return True

    direction = direction.normalize()

    # Check intersection with each triangle
    for triangle in triangles:
        t = awpy.visibility.VisibilityChecker._ray_triangle_intersection(None, start_vec, direction, triangle)
        if t is not None and t <= distance:
            return False

    return True


class TestVisibility:
    """Tests the Awpy calculation functions."""

    @pytest.fixture(autouse=True)
    def setup_runner(self):
        """Setup CLI runner."""
        self.runner = CliRunner()
        self.runner.invoke(awpy.cli.get, ["usd", "de_dust2"])

    def test_basic_visibility(self):
        """Tests basic visibility for de_dust2."""
        de_dust2_tri = awpy.data.TRIS_DIR / "de_dust2.tri"
        tris = awpy.visibility.VisibilityChecker.read_tri_file(de_dust2_tri)
        vc = awpy.visibility.VisibilityChecker(path=de_dust2_tri)

        test_points = [
            # Structured as (point1, point2, expected_visibility)
            (
                (-651, -831, 179),  # t_spawn_pos_1
                (-992, -766, 181),  # t_spawn_pos_2
                True,
            ),
            (
                (-651, -831, 179),  # t_spawn_pos_1
                (15, 2168, -65),  # ct_spawn_pos
                False,
            ),
            (
                (-485.90, 1737.51, -60.28),  # mid_doors_ct
                (-489.97, 1532.02, -61.08),  # mid_doors_t
                False,
            ),
            (
                (-515.23, 2251.36, -55.76),  # ct_spawn_towards_b
                (1318.11, 2027.95, 62.41),  # long_a_near_site
                True,
            ),
            (
                (195.87492752075195, 2467.874755859375, -52.5000057220459),
                (-860.0001831054688, -733.0000610351562, 190.00000254313153),
                False,
            ),
        ]

        # Test both BVH and brute force methods
        for start, end, expected in test_points:
            bvh_result = vc.is_visible(start, end)
            brute_force_result = check_visibility_brute_force(start, end, tris)

            # Test visibility in both directions
            bvh_result_reverse = vc.is_visible(end, start)
            brute_force_result_reverse = check_visibility_brute_force(end, start, tris)

            # Assert all results match the expected outcome
            assert bvh_result == expected, f"BVH visibility from {start} to {end} failed"
            assert brute_force_result == expected, f"Brute force visibility from {start} to {end} failed"
            assert bvh_result == brute_force_result, f"BVH and brute force results differ for {start} to {end}"

            # Assert reverse direction matches
            assert bvh_result == bvh_result_reverse, f"BVH visibility not symmetric for {start} and {end}"
            assert brute_force_result == brute_force_result_reverse, (
                f"Brute force visibility not symmetric for {start} and {end}"
            )


class TestFlatTriReader:
    """Test the flat numpy triangle reader."""

    @pytest.fixture(autouse=True)
    def setup_runner(self):
        self.runner = CliRunner()
        self.runner.invoke(awpy.cli.get, ["usd", "de_dust2"])

    def test_flat_reader_matches_object_reader(self):
        """Flat reader produces same data as object-based reader."""
        tri_path = awpy.data.TRIS_DIR / "de_dust2.tri"
        tris_objects = awpy.visibility.VisibilityChecker.read_tri_file(tri_path)
        tris_flat = awpy.visibility.read_tri_flat(tri_path)

        assert tris_flat.shape == (len(tris_objects), 9)
        assert tris_flat.dtype == np.float64

        # Spot-check first and last triangle
        t0 = tris_objects[0]
        np.testing.assert_allclose(
            tris_flat[0],
            [t0.p1.x, t0.p1.y, t0.p1.z, t0.p2.x, t0.p2.y, t0.p2.z, t0.p3.x, t0.p3.y, t0.p3.z],
            atol=1e-5,
        )
        t_last = tris_objects[-1]
        np.testing.assert_allclose(
            tris_flat[-1],
            [t_last.p1.x, t_last.p1.y, t_last.p1.z, t_last.p2.x, t_last.p2.y, t_last.p2.z, t_last.p3.x, t_last.p3.y, t_last.p3.z],
            atol=1e-5,
        )


class TestFlatBVHBuild:
    """Test flat BVH construction."""

    def test_build_small_bvh(self):
        """Build a BVH from a small set of triangles and verify structure."""
        tris = np.array([
            [0,0,0, 1,0,0, 0,1,0],
            [5,5,5, 6,5,5, 5,6,5],
            [2,2,2, 3,2,2, 2,3,2],
        ], dtype=np.float64)

        nodes, children, tri_idx = awpy.visibility.build_flat_bvh(tris)

        assert nodes.shape[1] == 6
        assert children.shape[1] == 2
        assert len(tri_idx) == len(nodes)

        root_min = nodes[0, :3]
        root_max = nodes[0, 3:]
        assert root_min[0] <= 0.0
        assert root_max[0] >= 6.0

        leaf_count = np.sum(tri_idx >= 0)
        assert leaf_count == 3

    @pytest.fixture(autouse=True)
    def setup_runner(self):
        self.runner = CliRunner()
        self.runner.invoke(awpy.cli.get, ["usd", "de_dust2"])

    def test_build_dust2_bvh(self):
        """Build BVH for de_dust2 and verify basic properties."""
        tris = awpy.visibility.read_tri_flat(awpy.data.TRIS_DIR / "de_dust2.tri")
        nodes, children, tri_idx = awpy.visibility.build_flat_bvh(tris)

        leaf_count = int(np.sum(tri_idx >= 0))
        assert leaf_count == len(tris)

        internal_mask = tri_idx < 0
        assert np.all(children[internal_mask, 0] >= 0)
        assert np.all(children[internal_mask, 1] >= 0)


class TestNumbaTraversal:
    """Test numba-JIT'd ray traversal matches original."""

    @pytest.fixture(autouse=True)
    def setup_runner(self):
        self.runner = CliRunner()
        self.runner.invoke(awpy.cli.get, ["usd", "de_dust2"])

    def test_numba_matches_original_visibility(self):
        """Numba traversal produces identical results to original BVH."""
        tri_path = awpy.data.TRIS_DIR / "de_dust2.tri"

        # Original checker
        vc_original = awpy.visibility.VisibilityChecker(path=tri_path)

        # Flat BVH checker
        tris_flat = awpy.visibility.read_tri_flat(tri_path)
        nodes, children, tri_idx = awpy.visibility.build_flat_bvh(tris_flat)

        test_pairs = [
            ((-651, -831, 179), (-992, -766, 181), True),
            ((-651, -831, 179), (15, 2168, -65), False),
            ((-485.90, 1737.51, -60.28), (-489.97, 1532.02, -61.08), False),
            ((-515.23, 2251.36, -55.76), (1318.11, 2027.95, 62.41), True),
            ((195.87, 2467.87, -52.50), (-860.00, -733.00, 190.00), False),
        ]

        for start, end, expected in test_pairs:
            original_result = vc_original.is_visible(start, end)
            assert original_result == expected, f"Original failed for {start}->{end}"

            numba_result = awpy.visibility.is_visible_flat(
                nodes, children, tri_idx, tris_flat,
                np.array(start, dtype=np.float64),
                np.array(end, dtype=np.float64),
            )
            assert numba_result == expected, f"Numba failed for {start}->{end}"

    def test_numba_symmetry(self):
        """Visibility is symmetric (A->B == B->A)."""
        tri_path = awpy.data.TRIS_DIR / "de_dust2.tri"
        tris_flat = awpy.visibility.read_tri_flat(tri_path)
        nodes, children, tri_idx = awpy.visibility.build_flat_bvh(tris_flat)

        pairs = [
            ((-651, -831, 179), (-992, -766, 181)),
            ((-651, -831, 179), (15, 2168, -65)),
            ((-485.90, 1737.51, -60.28), (-489.97, 1532.02, -61.08)),
        ]

        for start, end in pairs:
            fwd = awpy.visibility.is_visible_flat(
                nodes, children, tri_idx, tris_flat,
                np.array(start, dtype=np.float64),
                np.array(end, dtype=np.float64),
            )
            rev = awpy.visibility.is_visible_flat(
                nodes, children, tri_idx, tris_flat,
                np.array(end, dtype=np.float64),
                np.array(start, dtype=np.float64),
            )
            assert fwd == rev, f"Asymmetric for {start}<->{end}"


class TestBVHCache:
    """Test BVH save/load cycle."""

    @pytest.fixture(autouse=True)
    def setup_runner(self):
        self.runner = CliRunner()
        self.runner.invoke(awpy.cli.get, ["usd", "de_dust2"])

    def test_save_load_roundtrip(self, tmp_path):
        """Saved BVH loads back identically."""
        tri_path = awpy.data.TRIS_DIR / "de_dust2.tri"
        tris = awpy.visibility.read_tri_flat(tri_path)
        nodes, children, tri_idx = awpy.visibility.build_flat_bvh(tris)

        bvh_path = tmp_path / "de_dust2.bvh"
        awpy.visibility.save_flat_bvh(bvh_path, nodes, children, tri_idx)

        nodes2, children2, tri_idx2 = awpy.visibility.load_flat_bvh(bvh_path)

        np.testing.assert_array_equal(nodes, nodes2)
        np.testing.assert_array_equal(children, children2)
        np.testing.assert_array_equal(tri_idx, tri_idx2)

    def test_cached_checker_matches_fresh(self, tmp_path):
        """VisibilityChecker with cached BVH produces same results."""
        import shutil
        tri_path = awpy.data.TRIS_DIR / "de_dust2.tri"

        tmp_tri = tmp_path / "de_dust2.tri"
        shutil.copy(tri_path, tmp_tri)

        # First load: builds BVH and caches it
        vc1 = awpy.visibility.VisibilityChecker(path=tmp_tri)
        bvh_file = tmp_tri.with_suffix(".bvh")
        assert bvh_file.exists(), "BVH cache file was not created"

        # Second load: should use cache
        vc2 = awpy.visibility.VisibilityChecker(path=tmp_tri)

        test_points = [
            ((-651, -831, 179), (-992, -766, 181)),
            ((-651, -831, 179), (15, 2168, -65)),
        ]
        for start, end in test_points:
            assert vc1.is_visible(start, end) == vc2.is_visible(start, end)


class TestBatchVisibility:
    """Test batch visibility checking."""

    @pytest.fixture(autouse=True)
    def setup_runner(self):
        self.runner = CliRunner()
        self.runner.invoke(awpy.cli.get, ["usd", "de_dust2"])

    def test_batch_matches_individual(self):
        """Batch results match individual is_visible calls."""
        tri_path = awpy.data.TRIS_DIR / "de_dust2.tri"
        vc = awpy.visibility.VisibilityChecker(path=tri_path)

        starts = np.array([
            [-651, -831, 179],
            [-651, -831, 179],
            [-485.90, 1737.51, -60.28],
            [-515.23, 2251.36, -55.76],
        ], dtype=np.float64)
        ends = np.array([
            [-992, -766, 181],
            [15, 2168, -65],
            [-489.97, 1532.02, -61.08],
            [1318.11, 2027.95, 62.41],
        ], dtype=np.float64)

        batch_results = vc.is_visible_batch(starts, ends)
        individual_results = [vc.is_visible(tuple(s), tuple(e)) for s, e in zip(starts, ends)]

        np.testing.assert_array_equal(batch_results, individual_results)
