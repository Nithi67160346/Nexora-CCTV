import unittest
from unittest.mock import patch

from web.violence_clip import build_model, checkpoint_layers


def state_keys(layers):
    return {f'lstm.{part}_l{layer}': None for layer in range(layers)
            for part in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')}


class ViolenceV3Tests(unittest.TestCase):
    def test_supplied_v3_and_legacy_layers_are_detected(self):
        self.assertEqual(checkpoint_layers(state_keys(2)), 2)
        self.assertEqual(checkpoint_layers(state_keys(1)), 1)

    def test_partial_or_unsupported_checkpoint_is_rejected(self):
        partial = state_keys(2)
        partial.pop('lstm.bias_hh_l1')
        for state in (partial, state_keys(3), {}, {'lstm.weight_ih_l0_reverse': None}, {0: None}, []):
            with self.subTest(state=state), self.assertRaises(ValueError):
                checkpoint_layers(state)

    def test_build_uses_supplied_weights_without_imagenet_download(self):
        import torch.nn as nn
        # Keep the constructor test small; real supplied weights get a separate parity probe.
        with patch('torchvision.models.resnet18', return_value=nn.Sequential(nn.Identity())) as resnet:
            model = build_model(2)
            resnet.assert_called_once_with(weights=None)
        self.assertEqual(model.lstm.num_layers, 2)
        self.assertEqual(model.lstm.dropout, .3)
        self.assertEqual(model.dropout.p, .5)
        self.assertEqual(model.fc.out_features, 2)
        model.eval()
        self.assertFalse(model.dropout.training)
