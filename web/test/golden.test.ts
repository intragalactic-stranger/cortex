// Parity with the Python memcalc: the hand-calculated golden vectors in serve/tests/fixtures/golden.json,
// asserted with the same checks and tolerances as serve/tests/test_golden.py.
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { estimate, fixedStateBytes, kvBytesPerToken, seqBytes, weightsBytes } from "../src/lib/memcalc/calc.ts";
import { getDevice } from "../src/lib/memcalc/devices.ts";
import { storedParams } from "../src/lib/memcalc/model.ts";
import { loadDevices, loadFixtureModel, readJson, repoPath } from "../src/lib/repo.ts";

interface Golden {
  cases: { id: string; model: string; kv_dtype: string; ctx: number; expect: Record<string, number> }[];
  estimates: {
    id: string;
    model: string;
    device: string;
    dtype: string;
    kv_dtype: string;
    ctx: number;
    util: number;
    expect: Record<string, number | string>;
  }[];
}

const GOLDEN = readJson<Golden>(repoPath("serve", "tests", "fixtures", "golden.json"));
const DEVICES = loadDevices();

/** pytest.approx(expected, rel=tol) */
function approx(actual: number, expected: number, rel: number): void {
  assert.ok(Math.abs(actual - expected) <= rel * Math.abs(expected), `${actual} != ${expected} ± ${rel * 100}%`);
}

describe("golden memory cases", () => {
  for (const c of GOLDEN.cases) {
    it(c.id, () => {
      const s = loadFixtureModel(c.model);
      const exp = c.expect;
      assert.equal(seqBytes(s, c.ctx, c.kv_dtype), exp["seq_bytes"]);
      if ("kv_bytes_per_token" in exp) assert.equal(kvBytesPerToken(s, c.kv_dtype), exp["kv_bytes_per_token"]);
      if ("fixed_state_bytes" in exp) assert.equal(fixedStateBytes(s), exp["fixed_state_bytes"]);
      if ("params_stored" in exp) assert.equal(storedParams(s.params), exp["params_stored"]);
    });
  }
});

describe("golden estimates", () => {
  for (const c of GOLDEN.estimates) {
    it(c.id, () => {
      const s = loadFixtureModel(c.model);
      const exp = c.expect;
      const est = estimate(s, getDevice(DEVICES, c.device), {
        weightFmt: c.dtype,
        kvFmt: c.kv_dtype,
        ctx: c.ctx,
        util: c.util,
      });
      if ("weights_bytes" in exp) assert.equal(weightsBytes(s, c.dtype), exp["weights_bytes"]);
      if ("max_concurrent_seqs" in exp) assert.equal(est.max_concurrent_seqs, exp["max_concurrent_seqs"]);
      if ("decode_ceiling_tok_s" in exp) {
        assert.ok(est.decode_ceiling_tok_s !== null);
        approx(est.decode_ceiling_tok_s, Number(exp["decode_ceiling_tok_s"]), Number(exp["tolerance"]));
      }
    });
  }
  it("covers every golden case", () => {
    assert.ok(GOLDEN.cases.length >= 8 && GOLDEN.estimates.length >= 3);
  });
});
