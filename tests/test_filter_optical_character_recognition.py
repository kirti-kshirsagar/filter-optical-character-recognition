#!/usr/bin/env python

import logging
import multiprocessing
import os
import sys
import tempfile
import unittest
import json
import cv2
import numpy as np
from openfilter.filter_runtime.filter import Frame

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from filter_optical_character_recognition.filter import (
    FilterOpticalCharacterRecognition,
    FilterOpticalCharacterRecognitionConfig,
    OCREngine,
)

logger = logging.getLogger(__name__)

logger.setLevel(int(getattr(logging, (os.getenv("LOG_LEVEL") or "INFO").upper())))

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
LOG_LEVEL = logger.getEffectiveLevel()


class TestFilterOpticalCharacterRecognition(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_file = os.path.join(self.temp_dir.name, "output.json")
        # Create the output file directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_test_frame(self, text=None, frame_id=1, skip_ocr=False):
        """Helper method to create a test frame with optional text and metadata."""
        image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        if text:
            cv2.putText(
                image,
                text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return {
            "main": Frame(image, {"meta": {"id": frame_id}}, "BGR"),
            "test_frame": Frame(
                image, {"meta": {"id": frame_id, "skip_ocr": skip_ocr}}, "BGR"
            ),
        }

    def test_setup_with_tesseract(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="tesseract", output_json_path=self.output_file
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        self.assertEqual(filter_app.ocr_engine, OCREngine.TESSERACT)
        self.assertTrue(os.path.exists(self.output_file))
        self.assertIsNotNone(filter_app.output_file)

        self.assertFalse(filter_app.output_file.closed)
        filter_app.shutdown()
        self.assertTrue(filter_app.output_file.closed)

    def test_setup_with_easyocr(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="easyocr", output_json_path=self.output_file
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        self.assertEqual(filter_app.ocr_engine, OCREngine.EASYOCR)
        self.assertIsNotNone(filter_app.easyocr_reader)
        self.assertTrue(os.path.exists(self.output_file))
        self.assertIsNotNone(filter_app.output_file)

        self.assertFalse(filter_app.output_file.closed)
        filter_app.shutdown()
        self.assertTrue(filter_app.output_file.closed)

    def test_process_with_tesseract(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="tesseract",
            ocr_language=["en", "eng"],
            output_json_path=self.output_file,
            tesseract_cmd=os.path.abspath("bin/tesseract/tesseract.AppImage"),
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        frame = self.create_test_frame("Open your EYE", 1)
        filter_app.process(frame)
        filter_app.shutdown()

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            result = json.loads(lines[-1])
            self.assertEqual(result["frame_id"], 1)
            self.assertIn("Open your EYE", result["texts"])
            self.assertIn("ocr_confidence", result)
            self.assertIsInstance(result["ocr_confidence"], (int, float))
            self.assertGreaterEqual(result["ocr_confidence"], 0.0)
            self.assertLessEqual(result["ocr_confidence"], 100.0)

    def test_process_with_easyocr(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="easyocr", output_json_path=self.output_file
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        frame = self.create_test_frame("Open your EYE", 2)
        filter_app.process(frame)
        filter_app.shutdown()

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            result = json.loads(lines[-1])
            self.assertEqual(result["frame_id"], 2)
            self.assertIn("Open your EYE", result["texts"])
            self.assertIn("ocr_confidence", result)
            self.assertIsInstance(result["ocr_confidence"], (int, float))

    def test_process_empty_frame_tesseract(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="tesseract",
            ocr_language=["en", "eng"],
            output_json_path=self.output_file,
            tesseract_cmd=os.path.abspath("bin/tesseract/tesseract.AppImage"),
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        frame = self.create_test_frame(None, 3)
        filter_app.process(frame)
        filter_app.shutdown()

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            result = json.loads(lines[-1])
            self.assertEqual(result["frame_id"], 3)
            self.assertEqual(result["texts"], [])
            self.assertIn("ocr_confidence", result)
            self.assertEqual(result["ocr_confidence"], 0.0)

    def test_process_empty_frame_easyocr(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="easyocr", output_json_path=self.output_file
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        frame = self.create_test_frame(None, 4)
        filter_app.process(frame)
        filter_app.shutdown()

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            result = json.loads(lines[-1])
            self.assertEqual(result["frame_id"], 4)
            self.assertEqual(result["ocr_confidence"], 0.0)
            self.assertEqual(result["texts"], [])

    def test_invalid_ocr_engine(self):
        with self.assertRaises(ValueError):
            config = FilterOpticalCharacterRecognitionConfig(
                ocr_engine="INVALID_ENGINE", output_json_path=self.output_file
            )
            FilterOpticalCharacterRecognition(config)

    def test_output_file_writing_multiple_frames(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="tesseract",
            ocr_language=["en", "eng"],
            output_json_path=self.output_file,
            tesseract_cmd=os.path.abspath("bin/tesseract/tesseract.AppImage"),
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        # Create multiple frames with different text
        texts = ["Frame One", "Frame Two", "Frame Three"]
        for i, text in enumerate(texts, start=1):
            frame = self.create_test_frame(text, i)
            filter_app.process(frame)

        filter_app.shutdown()

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(texts))
            for i, line in enumerate(lines):
                result = json.loads(line)
                self.assertEqual(result["frame_id"], i + 1)
                self.assertIn(texts[i], result["texts"])

    def test_invalid_output_file_path(self):
        invalid_path = "/invalid/path/output.json"
        with self.assertRaises(Exception):
            config = FilterOpticalCharacterRecognitionConfig(
                ocr_engine="tesseract", output_json_path=invalid_path
            )
            filter_app = FilterOpticalCharacterRecognition(config)
            filter_app.setup(filter_app.normalize_config(config))

    def test_output_file_integrity(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="tesseract",
            ocr_language=["en", "eng"],
            output_json_path=self.output_file,
            tesseract_cmd=os.path.abspath("bin/tesseract/tesseract.AppImage"),
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        texts = ["First Frame", "Second Frame", "Third Frame"]
        for i, text in enumerate(texts, start=1):
            frame = self.create_test_frame(text, i)
            filter_app.process(frame)

        filter_app.shutdown()

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(texts))
            for line in lines:
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    self.fail("Output file contains invalid JSON.")

    def test_output_json_appends_correctly(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="tesseract",
            ocr_language=["en", "eng"],
            output_json_path=self.output_file,
            tesseract_cmd=os.path.abspath("bin/tesseract/tesseract.AppImage"),
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        texts_batch1 = ["Batch1 Frame1", "Batch1 Frame2"]
        for i, text in enumerate(texts_batch1, start=1):
            frame = self.create_test_frame(text, i)
            filter_app.process(frame)

        texts_batch2 = ["Batch2 Frame1", "Batch2 Frame2"]
        for i, text in enumerate(texts_batch2, start=3):
            frame = self.create_test_frame(text, i)
            filter_app.process(frame)

        filter_app.shutdown()

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(texts_batch1) + len(texts_batch2))
            for i, line in enumerate(lines):
                result = json.loads(line)
                self.assertEqual(result["frame_id"], i + 1)
                self.assertIn(result["texts"][0], texts_batch1 + texts_batch2)

    def test_initialize_with_defaults(self):
        """Test initializing the OCR filter with default settings."""
        config = FilterOpticalCharacterRecognitionConfig()
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        self.assertEqual(filter_app.ocr_engine, OCREngine.EASYOCR)
        self.assertEqual(filter_app.output_json_path, "./output/ocr_results.json")
        self.assertEqual(filter_app.debug, False)
        self.assertIn(filter_app.language, [["en"], ["eng"]])

        self.assertIsNotNone(filter_app.easyocr_reader)

        filter_app.shutdown()

    def test_skip_ocr_true(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="easyocr", output_json_path=self.output_file
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        frame = self.create_test_frame("Open your EYE", 1, skip_ocr=True)
        filter_app.process(frame)
        filter_app.shutdown()

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 0)  # no output

    def test_skip_ocr_false(self):
        config = FilterOpticalCharacterRecognitionConfig(
            ocr_engine="easyocr", output_json_path=self.output_file
        )
        filter_app = FilterOpticalCharacterRecognition(config)
        filter_app.setup(filter_app.normalize_config(config))

        frame = self.create_test_frame("Open your EYE", 2, skip_ocr=False)
        filter_app.process(frame)
        filter_app.shutdown()

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            result = json.loads(lines[-1])
            self.assertEqual(result["frame_id"], 2)
            self.assertIn("Open your EYE", result["texts"])


try:
    multiprocessing.set_start_method("spawn")  # CUDA doesn't like fork()
except Exception:
    pass

if __name__ == "__main__":
    unittest.main()
