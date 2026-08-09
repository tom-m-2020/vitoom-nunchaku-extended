import importlib.util
import pathlib
import unittest
import weakref

import torch


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "nunchaku/models/transformers/flux2_attention_callbacks.py"
)
SPEC = importlib.util.spec_from_file_location("flux2_attention_callbacks_test", MODULE_PATH)
CALLBACKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALLBACKS)


def metadata(block_type="double", block_index=3, refs=(5, 7)):
    text = 9
    generated = 11
    image = generated + sum(refs)
    if block_type == "double":
        padded_text, padded_image = 256, 256
        packed = 512
    else:
        padded_text, packed = text, 256
        padded_image = packed - text
    return CALLBACKS.Flux2AttentionInvocation(
        block_type=block_type,
        block_index=block_index,
        text_token_count=text,
        generated_token_count=generated,
        reference_token_counts=refs,
        logical_image_token_count=image,
        padded_text_token_count=padded_text,
        padded_image_token_count=padded_image,
        packed_sequence_length=packed,
        batch_size=1,
        head_count=2,
        head_dimension=4,
    )


class Flux2AttentionCallbackTests(unittest.TestCase):
    def tensors(self):
        return tuple(torch.ones((1, 2, 8, 4)) * value for value in (1, 2, 3))

    def test_capability_and_metadata_layouts(self):
        self.assertEqual(CALLBACKS.FLUX2_ATTENTION_CALLBACK_API_VERSION, 1)
        double = metadata("double", 4, (5, 7))
        single = metadata("single", 9, ())
        self.assertEqual((double.block_type, double.block_index), ("double", 4))
        self.assertEqual((single.block_type, single.block_index), ("single", 9))
        self.assertEqual(double.reference_token_counts, (5, 7))
        self.assertEqual(single.reference_token_counts, ())
        self.assertEqual(double.logical_image_token_count, 23)
        self.assertEqual(double.packed_sequence_length, 512)
        self.assertNotEqual(double.logical_image_token_count, double.padded_image_token_count)

    def test_no_callbacks_preserves_identity_and_values(self):
        q, k, v = self.tensors()
        result = CALLBACKS.run_pre_attention_callbacks((), q, k, v, metadata())
        self.assertEqual(tuple(map(id, result)), (id(q), id(k), id(v)))
        out = torch.arange(64.0).reshape(1, 8, 8)
        self.assertIs(
            CALLBACKS.run_post_attention_callbacks((), out, metadata()), out
        )

    def test_pre_order_in_place_visibility_and_valid_replacement(self):
        seen = []
        def first(q, k, v, info):
            seen.append(("first", float(k[0, 0, 0, 0])))
            k.mul_(2)
        def second(q, k, v, info):
            seen.append(("second", float(k[0, 0, 0, 0])))
            return q.clone(), k.clone(), v.clone()
        q, k, v = self.tensors()
        q2, k2, v2 = CALLBACKS.run_pre_attention_callbacks(
            (first, second), q, k, v, metadata()
        )
        self.assertEqual(seen, [("first", 2.0), ("second", 4.0)])
        self.assertIsNot(q2, q)
        self.assertTrue(torch.equal(k2, torch.full_like(k2, 4)))

    def test_pre_replacement_validation(self):
        q, k, v = self.tensors()
        bad = (
            lambda q, k, v, info: (q[:, :, :-1], k, v),
            lambda q, k, v, info: (q.double(), k, v),
            lambda q, k, v, info: (q.transpose(-1, -2), k, v),
        )
        for callback, error in zip(bad, (ValueError, TypeError, ValueError)):
            with self.subTest(callback=callback), self.assertRaises(error):
                CALLBACKS.run_pre_attention_callbacks((callback,), q, k, v, metadata())
        if torch.device("meta") != q.device:
            with self.assertRaises(ValueError):
                CALLBACKS.run_pre_attention_callbacks(
                    (lambda q, k, v, info: (torch.empty_like(q, device="meta"), k, v),),
                    q, k, v, metadata()
                )

    def test_malformed_callback_container_and_result(self):
        for value in ([], (object(),)):
            with self.assertRaises(TypeError):
                CALLBACKS.get_attention_callbacks(
                    {CALLBACKS.PRE_ATTENTION_CALLBACKS_KEY: value}
                )
        q, k, v = self.tensors()
        with self.assertRaises(TypeError):
            CALLBACKS.run_pre_attention_callbacks(
                (lambda *args: q,), q, k, v, metadata()
            )

    def test_post_order_mutation_replacement_and_validation(self):
        seen = []
        def first(out, info):
            seen.append(float(out[0, 0, 0]))
            out.add_(3)
        def second(out, info):
            seen.append(float(out[0, 0, 0]))
            return out.clone().mul_(2)
        out = torch.ones((1, 8, 8))
        result = CALLBACKS.run_post_attention_callbacks(
            (first, second), out, metadata()
        )
        self.assertEqual(seen, [1.0, 4.0])
        self.assertTrue(torch.equal(result, torch.full_like(result, 8)))
        for callback, error in (
            (lambda out, info: out[:, :-1], ValueError),
            (lambda out, info: out.double(), TypeError),
            (lambda out, info: out.transpose(1, 2), ValueError),
        ):
            with self.assertRaises(error):
                CALLBACKS.run_post_attention_callbacks((callback,), out, metadata())

    def test_exception_aborts_and_helper_retains_no_tensors(self):
        q, k, v = self.tensors()
        q_ref = weakref.ref(q)
        def fail(*args):
            raise RuntimeError("intentional")
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            CALLBACKS.run_pre_attention_callbacks((fail,), q, k, v, metadata())
        del q, k, v
        self.assertIsNone(q_ref())
        self.assertFalse(
            any(torch.is_tensor(value) for value in vars(CALLBACKS).values())
        )


if __name__ == "__main__":
    unittest.main()
