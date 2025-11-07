"""
Unit Tests for Multimodal Semantic Search Framework

These tests verify the functionality of all components without requiring
external dependencies like trained models or large datasets.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_import_encoders(self):
        """Test encoder imports."""
        try:
            from semantic_search.encoders import (
                TextEncoder, ImageEncoder, PanelEncoder,
                CharacterEncoder, ReadingOrderEncoder
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import encoders: {e}")
    
    def test_import_models(self):
        """Test model imports."""
        try:
            from semantic_search.models import MultimodalSemanticModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import models: {e}")
    
    def test_import_search_engine(self):
        """Test search engine import."""
        try:
            from semantic_search import SemanticSearchEngine
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import search engine: {e}")
    
    def test_import_utils(self):
        """Test utility imports."""
        try:
            from semantic_search.utils import (
                DocumentPreprocessor, ContrastiveLoss,
                TripletLoss, MultimodalDataset, Trainer
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import utils: {e}")
    
    def test_import_config(self):
        """Test config imports."""
        try:
            from semantic_search.config import DEFAULT_CONFIG, get_config
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import config: {e}")


class TestConfiguration(unittest.TestCase):
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration exists."""
        from semantic_search.config import DEFAULT_CONFIG
        
        self.assertIsInstance(DEFAULT_CONFIG, dict)
        self.assertIn('embedding_dim', DEFAULT_CONFIG)
        self.assertIn('text_encoder', DEFAULT_CONFIG)
        self.assertIn('image_encoder', DEFAULT_CONFIG)
    
    def test_get_config(self):
        """Test config getter."""
        from semantic_search.config import get_config
        
        config = get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config['embedding_dim'], 512)
    
    def test_config_override(self):
        """Test config override."""
        from semantic_search.config import get_config
        
        override = {'embedding_dim': 768}
        config = get_config(override)
        self.assertEqual(config['embedding_dim'], 768)


class TestEncoders(unittest.TestCase):
    """Test encoder initialization."""
    
    def test_text_encoder_init(self):
        """Test TextEncoder initialization."""
        from semantic_search.encoders import TextEncoder
        
        config = {
            'vocab_size': 50000,
            'hidden_dim': 512,
            'num_layers': 6,
            'num_heads': 8,
            'ff_dim': 2048,
            'max_seq_length': 512,
            'dropout': 0.1,
            'embedding_dim': 512,
        }
        
        encoder = TextEncoder(config)
        self.assertIsNotNone(encoder)
        self.assertEqual(encoder.hidden_dim, 512)
    
    def test_image_encoder_init(self):
        """Test ImageEncoder initialization."""
        from semantic_search.encoders import ImageEncoder
        
        config = {
            'input_channels': 3,
            'hidden_dim': 512,
            'num_layers': 6,
            'num_heads': 8,
            'ff_dim': 2048,
            'patch_size': 16,
            'image_size': 224,
            'dropout': 0.1,
            'embedding_dim': 512,
        }
        
        encoder = ImageEncoder(config)
        self.assertIsNotNone(encoder)
        self.assertEqual(encoder.hidden_dim, 512)
    
    def test_panel_encoder_init(self):
        """Test PanelEncoder initialization."""
        from semantic_search.encoders import PanelEncoder
        
        config = {
            'hidden_dim': 512,
            'num_layers': 3,
            'num_heads': 8,
            'max_panels': 50,
            'dropout': 0.1,
            'embedding_dim': 512,
        }
        
        encoder = PanelEncoder(config)
        self.assertIsNotNone(encoder)
        self.assertEqual(encoder.hidden_dim, 512)
    
    def test_character_encoder_init(self):
        """Test CharacterEncoder initialization."""
        from semantic_search.encoders import CharacterEncoder
        
        config = {
            'char_vocab_size': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'num_heads': 8,
            'max_char_length': 1024,
            'style_dim': 8,
            'dropout': 0.1,
            'embedding_dim': 512,
        }
        
        encoder = CharacterEncoder(config)
        self.assertIsNotNone(encoder)
        self.assertEqual(encoder.hidden_dim, 512)
    
    def test_reading_order_encoder_init(self):
        """Test ReadingOrderEncoder initialization."""
        from semantic_search.encoders import ReadingOrderEncoder
        
        config = {
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'ff_dim': 2048,
            'max_elements': 100,
            'num_element_types': 20,
            'max_hierarchy_levels': 10,
            'dropout': 0.1,
            'embedding_dim': 512,
        }
        
        encoder = ReadingOrderEncoder(config)
        self.assertIsNotNone(encoder)
        self.assertEqual(encoder.hidden_dim, 512)


class TestModel(unittest.TestCase):
    """Test MultimodalSemanticModel."""
    
    def test_model_init(self):
        """Test model initialization."""
        from semantic_search.models import MultimodalSemanticModel
        from semantic_search.config import get_config
        
        config = get_config()
        model = MultimodalSemanticModel(config)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.embedding_dim, 512)
    
    def test_model_fusion_types(self):
        """Test different fusion types."""
        from semantic_search.models import MultimodalSemanticModel
        from semantic_search.config import get_config
        
        for fusion_type in ['concat', 'attention', 'gated']:
            config = get_config({'fusion_type': fusion_type})
            model = MultimodalSemanticModel(config)
            self.assertEqual(model.fusion_type, fusion_type)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_preprocessor_init(self):
        """Test DocumentPreprocessor initialization."""
        from semantic_search.utils import DocumentPreprocessor
        from semantic_search.config import get_config
        
        config = get_config()
        preprocessor = DocumentPreprocessor(config['preprocessing'])
        
        self.assertIsNotNone(preprocessor)
        self.assertIsNotNone(preprocessor.char_to_id)
    
    def test_contrastive_loss_init(self):
        """Test ContrastiveLoss initialization."""
        from semantic_search.utils import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(temperature=0.07)
        self.assertIsNotNone(loss_fn)
        self.assertEqual(loss_fn.temperature, 0.07)
    
    def test_triplet_loss_init(self):
        """Test TripletLoss initialization."""
        from semantic_search.utils import TripletLoss
        
        loss_fn = TripletLoss(margin=0.5)
        self.assertIsNotNone(loss_fn)
        self.assertEqual(loss_fn.margin, 0.5)


class TestSyntax(unittest.TestCase):
    """Test Python syntax of all files."""
    
    def test_train_script_syntax(self):
        """Test train.py syntax."""
        import py_compile
        try:
            py_compile.compile('train.py', doraise=True)
            self.assertTrue(True)
        except py_compile.PyCompileError as e:
            self.fail(f"Syntax error in train.py: {e}")
    
    def test_inference_script_syntax(self):
        """Test inference.py syntax."""
        import py_compile
        try:
            py_compile.compile('inference.py', doraise=True)
            self.assertTrue(True)
        except py_compile.PyCompileError as e:
            self.fail(f"Syntax error in inference.py: {e}")
    
    def test_example_script_syntax(self):
        """Test examples/basic_usage.py syntax."""
        import py_compile
        try:
            py_compile.compile('examples/basic_usage.py', doraise=True)
            self.assertTrue(True)
        except py_compile.PyCompileError as e:
            self.fail(f"Syntax error in examples/basic_usage.py: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImports))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestEncoders))
    suite.addTests(loader.loadTestsFromTestCase(TestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntax))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
