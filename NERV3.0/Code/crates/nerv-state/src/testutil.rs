#[derive(Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub(crate) fn bytes32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        while out.len() + 8 <= n {
            out.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        if out.len() < n {
            let tail = self.next_u64().to_le_bytes();
            out.extend_from_slice(&tail[..n - out.len()]);
        }
        out
    }
    for rec in &b.claims {
           let _ = log.claim(&transit_key(&rec.txid, &st.shard(), rec.spend_leg), hgt);
       }
       b.header.nct_root = nct.root();
       b.header.nullifier_root = nfs.root();
       b.header.transit_root = log.root();
       b.header.fee_total = FeeSats::from_u64(fee);
       b.header.ct_batch_hash =
           Hash256::concat(&CT_BATCH, &ct_sum(&resolved).unwrap().to_bytes());
       for r in &resolved {
           world.chain.insert((hgt.as_u64(), r.key), r.clone());
       }
   }


   /// A valid one-spend-leg block at the state's next height.
   pub(crate) fn valid_block(st: &ShardState, world: &mut World, seed: u64) -> (ShardBlock, TxId) {
       let mut rng = SplitMix64::new(seed ^ 0x5EED_0000);
       let shard = st.shard();
       let shell = TransactionShell {
           legs: vec![leg(
               &mut rng,
               shard,
               vec![h(&mut rng)],
               vec![(1_000_000_000, false), (2_000_000_000, false)],
               1000,
               5_000,
               st.nct().root().as_hash256(),
           )],
       };
       let txid = shell.txid().unwrap();
       let interval = TAU0 + st.height().as_u64() + 1;
       let tree = register_tau(world, interval, &[txid]);
       let sl = SettledLeg {
           shell,
           leg: LegIndex::FIRST,
           tau: tree.witness(tree.position(&txid).unwrap()).unwrap(),
           siblings: vec![],
       };
       let mut b = block(shard, interval, vec![sl]);
       seal(st, &mut b, world);
       (b, txid)
   }


   /// A chain of `n` valid blocks; `states[i]` is the state after `i`
   /// blocks (`states[0]` = genesis). Distinct seeds give distinct chains.
   pub(crate) fn build_chain(
       shard: ShardId,
       n: u64,
       seed: u64,
   ) -> (Vec<ShardBlock>, Vec<ShardState>, World) {
       let mut world = World::default();
       let mut st = ShardState::genesis(shard, params());
       let mut blocks = Vec::new();
       let mut states = vec![st.clone()];
       for _ in 0..n {
           let (b, _) = valid_block(&st, &mut world, seed ^ (states.len() as u64));
           let (s, _) = apply_block(st, &b, &world, &world).unwrap();
           st = s;
           blocks.push(b);
           states.push(st.clone());
       }
       (blocks, states, world)
   }

