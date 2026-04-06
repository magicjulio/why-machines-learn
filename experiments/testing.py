from __future__ import annotations

import unittest

from app import create_app


def make_image(value: int) -> dict[str, object]:
    pixels = [value] * (28 * 28)
    return {"width": 28, "height": 28, "pixels": pixels}


class HopfieldApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = create_app()
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_hopfield_gallery_rejects_more_than_five_images(self):
        payload = {
            "mode": "gallery",
            "uploaded_store": [make_image(255) for _ in range(6)],
            "included_indices": [0],
            "selected_index": 0,
            "query_image": make_image(255),
        }

        response = self.client.post("/api/hopfield/run", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("at most 5 images", data["error"])

    def test_hopfield_gallery_allows_empty_included_indices_as_default_all(self):
        payload = {
            "mode": "gallery",
            "uploaded_store": [make_image(0), make_image(255)],
            "included_indices": [],
            "selected_index": 0,
            "query_image": make_image(255),
        }

        response = self.client.post("/api/hopfield/run", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["params"]["mode"], "gallery")
        self.assertEqual(len(data["stored"]["labels"]), 2)

    def test_hopfield_gallery_rejects_out_of_bounds_selected_index(self):
        payload = {
            "mode": "gallery",
            "uploaded_store": [make_image(0), make_image(255)],
            "included_indices": [0, 1],
            "selected_index": 7,
            "query_image": make_image(255),
        }

        response = self.client.post("/api/hopfield/run", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("out of bounds", data["error"])

    def test_hopfield_gallery_with_manual_query_classifies(self):
        payload = {
            "mode": "gallery",
            "uploaded_store": [make_image(0), make_image(255), make_image(0)],
            "included_indices": [0, 1],
            "selected_index": 1,
            "query_image": make_image(255),
            "noise_ratio": 0.2,
            "seed": 11,
        }

        response = self.client.post("/api/hopfield/run", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["params"]["mode"], "gallery")
        self.assertEqual(data["dataset"]["query_source"], "manual")
        self.assertTrue(set(data["stored"].keys()) >= {"labels", "image_shape", "images"})
        self.assertEqual(len(data["stored"]["labels"]), 2)
        self.assertIn(data["summary"]["pred_label"], data["stored"]["labels"])

    def test_hopfield_upload_mode_still_works(self):
        payload = {
            "mode": "upload",
            "uploaded_image": make_image(255),
            "noise_ratio": 0.1,
            "seed": 3,
        }

        response = self.client.post("/api/hopfield/run", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["params"]["mode"], "upload")
        self.assertEqual(data["dataset"]["source"], "user upload")
        self.assertEqual(data["summary"]["pred_label"], "upload")

    def test_nearest_label_returns_value_from_memory_set(self):
        payload = {
            "mode": "gallery",
            "uploaded_store": [make_image(255), make_image(0)],
            "included_indices": [0, 1],
            "selected_index": 0,
            "query_image": make_image(255),
        }

        response = self.client.post("/api/hopfield/run", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn(data["summary"]["pred_label"], {"img_1", "img_2"})
        self.assertIsInstance(data["summary"]["pred_distance"], int)

    def test_payload_images_are_normalized_to_expected_shape(self):
        payload = {
            "mode": "gallery",
            "uploaded_store": [{"width": 14, "height": 14, "pixels": [255] * (14 * 14)}],
            "included_indices": [0],
            "selected_index": 0,
            "query_image": {"width": 14, "height": 14, "pixels": [255] * (14 * 14)},
        }

        response = self.client.post("/api/hopfield/run", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["stored"]["image_shape"], [40, 40])

    def test_query_source_random_when_query_image_omitted(self):
        payload = {
            "mode": "gallery",
            "uploaded_store": [make_image(255), make_image(0)],
            "included_indices": [0, 1],
            "selected_index": 0,
            "noise_ratio": 0.2,
            "seed": 1,
        }

        response = self.client.post("/api/hopfield/run", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["dataset"]["query_source"], "random")

    def test_gallery_included_indices_must_be_list(self):
        payload = {
            "mode": "gallery",
            "uploaded_store": [make_image(255)],
            "included_indices": "0",
            "selected_index": 0,
            "query_image": make_image(255),
        }

        response = self.client.post("/api/hopfield/run", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("must be a list", data["error"])


if __name__ == "__main__":
    unittest.main()
