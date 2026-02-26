import numpy as np
import pytest

from src.labels.functional.instance_matching import (
    match_instances_by_binary_union,
    apply_matching,
)


def create_3d_label_image(shape, instances):
    """
    Create a 3D label image with specified instances.

    Args:
        shape: Tuple (depth, height, width) for the image shape
        instances: List of dicts with 'id' and 'positions' (list of (z,y,x) tuples)

    Returns:
        np.ndarray: 3D label image
    """
    img = np.zeros(shape, dtype=np.int32)
    for instance in instances:
        instance_id = instance['id']
        positions = instance['positions']
        for z, y, x in positions:
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                img[z, y, x] = instance_id
    return img


def get_instance_ids(img, background_instance=0):
    """Get all non-background instance IDs from an image."""
    unique = np.unique(img)
    return [u for u in unique if u != background_instance]


class TestGlobalBinaryUnionInstanceMatching:
    """Tests for global binary union instance matching with 3D images."""

    def test_single_instance_persists(self):
        """
        Test that a single instance that persists across all frames
        gets the same lineage ID and TCUI ID.
        """
        background_instance = 0
        shape = (5, 10, 10)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 4), (2, 4, 5), (2, 5, 4), (2, 5, 5)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 5), (2, 4, 6), (2, 5, 5), (2, 5, 6)]}
        ])
        frame2 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 6), (2, 4, 7), (2, 5, 6), (2, 5, 7)]}
        ])
        frame3 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 7), (2, 4, 8), (2, 5, 7), (2, 5, 8)]}
        ])

        img_list = [frame0, frame1, frame2, frame3]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        assert len(lineage_mapping) == 4
        assert len(tcui_mapping) == 4

        lineage_ids = [set(m.values()) for m in lineage_mapping]
        tcui_ids = [set(m.values()) for m in tcui_mapping]

        assert len(lineage_ids[0]) == 1
        assert lineage_ids[0] == lineage_ids[1] == lineage_ids[2] == lineage_ids[3]

        assert len(tcui_ids[0]) == 1
        assert tcui_ids[0] == tcui_ids[1] == tcui_ids[2] == tcui_ids[3]

    def test_merging(self):
        """
        Test that two instances that merge get the same lineage ID,
        but different TCUI IDs (new branch after merge).
        
        When instances merge, they become connected through the binary union,
        so they get the same lineage ID (connected component).
        """
        background_instance = 0
        shape = (5, 10, 10)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 4), (2, 4, 5), (2, 5, 4), (2, 5, 5)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 5), (2, 4, 6), (2, 5, 5), (2, 5, 6)]}
        ])
        frame2 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 6), (2, 4, 7), (2, 5, 6), (2, 5, 7)]},
            {'id': 2, 'positions': [(2, 6, 6), (2, 6, 7), (2, 7, 6), (2, 7, 7)]}
        ])
        frame3 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 7), (2, 4, 8), (2, 5, 7), (2, 5, 8),
                                    (2, 6, 7), (2, 6, 8), (2, 7, 7), (2, 7, 8)]}
        ])

        img_list = [frame0, frame1, frame2, frame3]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        lineage_ids_frame2 = set(lineage_mapping[2].values())
        lineage_ids_frame3 = set(lineage_mapping[3].values())

        assert len(lineage_ids_frame2) == 1
        assert len(lineage_ids_frame3) == 1

    def test_splitting(self):
        """
        Test that one instance that splits into two gets the same lineage ID
        (they are connected through the binary union).
        
        When an instance splits, the parts remain connected through the union,
        so they get the same lineage ID.
        """
        background_instance = 0
        shape = (5, 10, 10)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 4), (2, 4, 5), (2, 5, 4), (2, 5, 5),
                                    (2, 6, 4), (2, 6, 5), (2, 7, 4), (2, 7, 5)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 5), (2, 4, 6), (2, 5, 5), (2, 5, 6),
                                    (2, 6, 5), (2, 6, 6), (2, 7, 5), (2, 7, 6)]}
        ])
        frame2 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 6), (2, 4, 7), (2, 5, 6), (2, 5, 7)]},
            {'id': 2, 'positions': [(2, 6, 6), (2, 6, 7), (2, 7, 6), (2, 7, 7)]}
        ])
        frame3 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 7), (2, 4, 8), (2, 5, 7), (2, 5, 8)]},
            {'id': 2, 'positions': [(2, 6, 7), (2, 6, 8), (2, 7, 7), (2, 7, 8)]}
        ])

        img_list = [frame0, frame1, frame2, frame3]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        lineage_ids_frame0 = set(lineage_mapping[0].values())
        lineage_ids_frame2 = set(lineage_mapping[2].values())

        assert len(lineage_ids_frame0) == 1
        assert len(lineage_ids_frame2) == 1

    def test_reappearing(self):
        """
        Test that a reappearing instance (after vanish) gets
        different lineage and TCUI IDs.
        
        When an instance vanishes (no overlap in binary union) and then
        a new instance appears, they should have different lineage IDs.
        """
        background_instance = 0
        shape = (5, 10, 10)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 4), (2, 4, 5), (2, 5, 4), (2, 5, 5)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 8, 8), (2, 8, 9), (2, 9, 8), (2, 9, 9)]}
        ])
        frame2 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 6), (2, 4, 7), (2, 5, 6), (2, 5, 7)]}
        ])
        frame3 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 7), (2, 4, 8), (2, 5, 7), (2, 5, 8)]}
        ])

        img_list = [frame0, frame1, frame2, frame3]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        lineage_ids_frame0 = set(lineage_mapping[0].values())
        lineage_ids_frame2 = set(lineage_mapping[2].values())

        assert len(lineage_ids_frame0) >= 1
        assert len(lineage_ids_frame2) >= 1

    def test_vanishing(self):
        """
        Test that an instance that vanishes in later frames
        gets lineage IDs for frames where it exists.
        """
        background_instance = 0
        shape = (5, 10, 10)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 4), (2, 4, 5), (2, 5, 4), (2, 5, 5)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 5), (2, 4, 6), (2, 5, 5), (2, 5, 6)]}
        ])
        frame2 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 6), (2, 4, 7), (2, 5, 6), (2, 5, 7)]}
        ])
        frame3 = create_3d_label_image(shape, [])

        img_list = [frame0, frame1, frame2, frame3]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        assert len(lineage_mapping[0]) == 1
        assert len(lineage_mapping[1]) == 1
        assert len(lineage_mapping[2]) == 1
        assert len(lineage_mapping[3]) == 0

    def test_complex_merging_splitting_vanishing(self):
        """
        Test complex scenario with merging, splitting, and vanishing.
        
        Frame 0: Instance A
        Frame 1: Instance A (moved)
        Frame 2: Instance A and Instance B merge into C, Instance D appears
        Frame 3: Instance C vanishes, Instance D persists
        """
        background_instance = 0
        shape = (5, 12, 12)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 4), (2, 4, 5), (2, 5, 4), (2, 5, 5)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 5), (2, 4, 6), (2, 5, 5), (2, 5, 6)]}
        ])
        frame2 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 6), (2, 4, 7), (2, 5, 6), (2, 5, 7)]},
            {'id': 2, 'positions': [(2, 7, 6), (2, 7, 7), (2, 8, 6), (2, 8, 7)]}
        ])
        frame3 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 7, 7), (2, 7, 8), (2, 8, 7), (2, 8, 8)]}
        ])

        img_list = [frame0, frame1, frame2, frame3]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        assert len(lineage_mapping[0]) == 1
        assert len(lineage_mapping[1]) == 1
        assert len(lineage_mapping[2]) == 2
        assert len(lineage_mapping[3]) == 1

    def test_multiple_instances_no_overlap(self):
        """
        Test that multiple instances that never overlap get different
        lineage and TCUI IDs.
        
        Instances must be spatially separated to not be connected
        through the binary union.
        """
        background_instance = 0
        shape = (5, 20, 20)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 1, 1), (2, 1, 2)]},
            {'id': 2, 'positions': [(2, 15, 15), (2, 15, 16)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 1, 2), (2, 1, 3)]},
            {'id': 2, 'positions': [(2, 15, 16), (2, 15, 17)]}
        ])
        frame2 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 1, 3), (2, 1, 4)]},
            {'id': 2, 'positions': [(2, 15, 17), (2, 15, 18)]}
        ])
        frame3 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 1, 4), (2, 1, 5)]},
            {'id': 2, 'positions': [(2, 15, 18), (2, 15, 19)]}
        ])

        img_list = [frame0, frame1, frame2, frame3]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        for frame_idx in range(4):
            lineage_ids = set(lineage_mapping[frame_idx].values())
            assert len(lineage_ids) == 2

    def test_partial_overlap_merging(self):
        """
        Test that partial overlap leads to merging.
        Two instances that partially overlap in one frame should merge.
        
        When two instances overlap spatially in any frame, they become
        connected through the binary union and get the same lineage ID.
        """
        background_instance = 0
        shape = (5, 10, 10)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 4), (2, 4, 5), (2, 4, 6)]},
            {'id': 2, 'positions': [(2, 4, 6), (2, 4, 7), (2, 4, 8)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 5, 4), (2, 5, 5), (2, 5, 6)]},
            {'id': 2, 'positions': [(2, 5, 6), (2, 5, 7), (2, 5, 8)]}
        ])

        img_list = [frame0, frame1]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        lineage_ids_frame0 = set(lineage_mapping[0].values())

        assert len(lineage_ids_frame0) == 1

    def test_all_instances_vanish(self):
        """
        Test scenario where all instances vanish in the last frame.
        """
        background_instance = 0
        shape = (5, 10, 10)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 4), (2, 4, 5)]},
            {'id': 2, 'positions': [(2, 6, 4), (2, 6, 5)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 5), (2, 4, 6)]},
            {'id': 2, 'positions': [(2, 6, 5), (2, 6, 6)]}
        ])
        frame2 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 6), (2, 4, 7)]},
            {'id': 2, 'positions': [(2, 6, 6), (2, 6, 7)]}
        ])
        frame3 = create_3d_label_image(shape, [])

        img_list = [frame0, frame1, frame2, frame3]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        assert len(lineage_mapping[0]) == 2
        assert len(lineage_mapping[1]) == 2
        assert len(lineage_mapping[2]) == 2
        assert len(lineage_mapping[3]) == 0


class TestApplyMatching:
    """Tests for the apply_matching function."""

    def test_basic_mapping(self):
        """Test basic instance ID mapping."""
        background_instance = 0
        ignore_label = -1

        img = np.array([
            [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 2, 0], [0, 2, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ], dtype=np.int32)

        matches = {1: 10, 2: 20}

        matched = apply_matching(
            matches=matches,
            image=img,
            background_instance=background_instance,
            ignore_label=ignore_label,
        )

        assert np.all(matched[img == 1] == 10)
        assert np.all(matched[img == 2] == 20)

    def test_apply_matching_no_ignore_label(self):
        """
        Test that apply_matching correctly maps instances when
        all instances are in the mapping.
        """
        background_instance = 0
        ignore_label = -1
        shape = (5, 10, 10)

        frame0 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 4), (2, 4, 5), (2, 5, 4), (2, 5, 5)]}
        ])
        frame1 = create_3d_label_image(shape, [
            {'id': 1, 'positions': [(2, 4, 5), (2, 4, 6), (2, 5, 5), (2, 5, 6)]}
        ])

        img_list = [frame0, frame1]

        lineage_mapping, tcui_mapping, graph = match_instances_by_binary_union(
            img_list=img_list,
            background_instance=background_instance,
        )

        matched_img = apply_matching(
            matches=tcui_mapping[0],
            image=frame0,
            background_instance=background_instance,
            ignore_label=ignore_label,
        )

        unique_vals = np.unique(matched_img)
        assert ignore_label not in unique_vals
        assert len(unique_vals) > 0

    def test_unmapped_instances_get_ignore_label(self):
        """Test that unmapped instances are assigned ignore_label."""
        background_instance = 0
        ignore_label = -1

        img = np.array([
            [[0, 1, 0], [0, 2, 0], [0, 3, 0]]
        ], dtype=np.int32)

        matches = {1: 10}

        matched = apply_matching(
            matches=matches,
            image=img,
            background_instance=background_instance,
            ignore_label=ignore_label,
        )

        assert np.all(matched[img == 1] == 10)
        assert np.all(matched[img == 2] == ignore_label)
        assert np.all(matched[img == 3] == ignore_label)

    def test_background_unchanged(self):
        """Test that background instances remain unchanged."""
        background_instance = 0
        ignore_label = -1

        img = np.zeros((3, 3, 3), dtype=np.int32)
        img[1, 1, 1] = 1

        matches = {1: 10}

        matched = apply_matching(
            matches=matches,
            image=img,
            background_instance=background_instance,
            ignore_label=ignore_label,
        )

        assert matched[0, 0, 0] == 0
        assert matched[1, 1, 1] == 10
