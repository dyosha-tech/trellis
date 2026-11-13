import unittest

import torch

from trellis2.watermark import (
    WatermarkPayload,
    embed_material_watermark,
    extract_material_watermark,
)


class MaterialWatermarkTests(unittest.TestCase):
    def setUp(self):
        self.coords = torch.stack((torch.arange(24000), torch.zeros(24000), torch.zeros(24000)), dim=1)
        self.attrs = torch.full((24000, 6), 0.5, dtype=torch.float32)
        self.layout = {"roughness": slice(4, 5)}
        self.payload = WatermarkPayload("Example Creator", "2026-10-22T12:00:00+00:00")

    def test_pack_round_trip(self):
        self.assertEqual(WatermarkPayload.unpack(self.payload.pack()), self.payload)

    def test_embedding_extracts_after_row_shuffle(self):
        marked = embed_material_watermark(self.coords, self.attrs, self.layout, self.payload)
        permutation = torch.randperm(len(self.coords), generator=torch.Generator().manual_seed(7))
        decoded = extract_material_watermark(self.coords[permutation], marked[permutation], self.layout)
        self.assertEqual(decoded, self.payload)
        self.assertTrue(torch.equal(self.attrs[:, :4], marked[:, :4]))
        self.assertTrue(torch.equal(self.attrs[:, 5:], marked[:, 5:]))

    def test_unmarked_attrs_fail_detection(self):
        with self.assertRaises(ValueError):
            extract_material_watermark(self.coords, self.attrs, self.layout)


if __name__ == "__main__":
    unittest.main()
