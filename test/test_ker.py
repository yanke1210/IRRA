import os
import tempfile
import unittest

import torch

from datasets.bases import ImageTextMLMDataset, tokenize
from model.build import IRRA
from model.objectives import compute_ot
from utils.simple_tokenizer import SimpleTokenizer, extract_keyword_token_spans


class KeywordSpanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = SimpleTokenizer()

    def test_keyword_maps_to_every_bpe_piece(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("unbelievable\n")
            keyword_file = handle.name
        try:
            caption = "A man wears unbelievable, shoes."
            spans = extract_keyword_token_spans(
                caption, self.tokenizer, keyword_file=keyword_file)
            keyword_ids = self.tokenizer.encode("unbelievable")
            self.assertEqual(spans, [(3, 3 + len(keyword_ids))])

            mapper = ImageTextMLMDataset.__new__(ImageTextMLMDataset)
            mapper.tokenizer = self.tokenizer
            caption_ids = tokenize(caption, self.tokenizer)
            token_spans = mapper._map_keyword_spans_to_tokens(spans, caption_ids)
            self.assertEqual(
                token_spans, [list(range(4, 4 + len(keyword_ids)))])
            self.assertEqual(caption_ids[token_spans[0]].tolist(), keyword_ids)
        finally:
            os.unlink(keyword_file)


class PartialOTTest(unittest.TestCase):
    def test_bpe_pieces_are_pooled_per_keyword(self):
        features = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [8.0, 8.0]]])
        groups = torch.tensor([[1, 1, 2]])
        pooled, mask = IRRA._pool_keyword_spans(features, groups)
        self.assertTrue(torch.equal(mask, torch.tensor([[True, True]])))
        self.assertTrue(torch.equal(
            pooled, torch.tensor([[[2.0, 2.0], [8.0, 8.0]]])))

    def test_negative_ranking_is_finite_and_backpropagates(self):
        keywords = torch.tensor([
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ], requires_grad=True)
        patches = torch.tensor([
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 1.0], [0.1, 0.9], [0.0, -1.0], [0.0, -1.0]],
        ], requires_grad=True)
        mask = torch.tensor([[True, False], [True, False]])
        loss = compute_ot(
            keywords, patches, keyword_mask=mask, pids=torch.tensor([0, 1]),
            partial_ratio=0.5, margin=0.2, sinkhorn_iterations=5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(keywords.grad)
        self.assertIsNotNone(patches.grad)

    def test_same_pid_batch_has_zero_loss(self):
        keywords = torch.randn(2, 2, 4, requires_grad=True)
        patches = torch.randn(2, 4, 4, requires_grad=True)
        loss = compute_ot(keywords, patches, pids=torch.tensor([3, 3]))
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
