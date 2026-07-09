Expert Review and Strategic Blueprint for Superhuman Catan-Zero
Executive Summary of Diagnostic Findings
An exhaustive analysis of the Catan-Zero project architecture, training methodology, and search mechanics reveals a system that has achieved commendable engineering milestones—specifically regarding CUDA Multi-Process Service (MPS) integration, Rust-based featurization, and rigorous bug eradication. However, the system is currently constrained by structural design choices that severely limit its asymptotic strength and sample efficiency. The plateau observed at "turn 4" is not an inherent limitation of the data or the search budget, but rather the mathematical consequence of outdated value-function optimization, rigid gating heuristics, and an oversimplified approach to imperfect-information planning.

The most critical vulnerabilities lie in the value head's fragility, the continuous training loop's promotion criteria, and the absence of a game-theoretic mechanism for handling hidden information. The recent deep reinforcement learning literature (2023–2026) offers explicit solutions to these exact pathologies. The transition from continuous scalar regression to categorical value classification will definitively resolve the value-drift phenomenon. Furthermore, adopting Exponential Moving Average (EMA) checkpointing alongside soft gating will unblock the self-play flywheel, allowing incremental improvements to compound seamlessly. Finally, the system must address the theoretical gap between perfect-information AlphaZero and the imperfect-information reality of Catan by integrating belief-aware search mechanisms, Information Set Monte Carlo Tree Search (IS-MCTS), or deterministic sampling techniques.   

To achieve the objective of building the premier Catan AI, the following ten strategic imperatives must be adopted:

Abolish Scalar Value Regression: The value head's fragility is an artifact of Mean Squared Error (MSE) regression. Transition immediately to HL-Gauss (Histogram Loss) or Two-Hot categorical cross-entropy to stabilize multi-epoch training and unblock parameter scaling.   

Dismantle the Rigid SPRT Gate: The +30 Elo pentanomial gate is starving the generation fleet. Transition to a continuous KataGo-style Exponential Moving Average (EMA) deployment, treating the external panel purely as a regression tripwire.   

Bridge the Imperfect Information Deficit: Masking hidden states prevents cheating but induces strategic blindness. Implement belief-state auxiliary heads or Perfect Information Monte Carlo (PIMC) determinization to map opponent hand distributions.   

Inject Architectural Equivariance: The dense 35M transformer wastes massive capacity relearning the hexagonal grid's topology. Explicitly encode D 
6
​
  dihedral symmetries via data augmentation and graph-biased attention.   

Refine Gumbel Search at Wide Roots: The current c_scale=0.03 bypasses mctx noise amplification but neuters the completed-Q backup. Adopt Variance-Aware Sequential Halving and policy-target pruning to handle 54-wide stochastic placement roots.   

Implement TD(λ) for Credit Assignment: Relying exclusively on a delayed terminal outcome over 600-move trajectories destroys credit assignment. Incorporate bootstrapped TD(λ) hybrid targets to densify the reward signal.   

Activate Opponent-Pool Self-Play: The flatline against external heuristic bots indicates self-play inbreeding. Deploy the built opponent-pool infrastructure, allocating 15-20% of generation games against historical checkpoints to force strategy generalization.   

Re-evaluate the 91M Parameter Scaling Probe: Once categorical value classification is implemented, the 91M model will no longer suffer epoch-2 explosion. Given the negligible 4% neural network latency cost per leaf, large-scale parameter expansion is a primary driver of asymptotic strength.   

Decouple Policy and Value Learning Rates: The policy network thrives on data reuse, while the value network overfits. Implement disparate learning rates or stop-gradient operations between the representation trunk and the value head to balance representation learning.   

Solve the Python/Rust Search Bottleneck: While outside the immediate algorithmic scope, parallelizing the selection loop to feed batched evaluations to the GPU is a mandatory prerequisite for scaling search depth to n=96 or n=128 without severe throughput degradation.

1. Systemic Vulnerabilities and Methodological Flaws
The Catan-Zero report meticulously documents several failure modes, most notably the value-head self-distillation drift and the search algorithm losing to its own raw policy. While the engineering mitigations applied (such as champion-initialization and c_scale attenuation) successfully stabilized the system, they treat the symptoms rather than the underlying mathematical pathologies.

1.1 The Value Head Fragility is a Regression Artifact
The recurring scientific theme noted in the Catan-Zero project—that the value head cannot tolerate revisiting the same distribution without catastrophic overfitting (manifesting as a 69% value error increase over six ungated rounds)—is a known flaw of MSE regression in deep reinforcement learning. The report attributes this to the value head being "fragile," which is a misdiagnosis of the loss function's fundamental inadequacy in stochastic, high-variance environments.

When a neural network is trained via MSE to predict a scalar value representing the expectation of a highly stochastic outcome (such as a Settlers of Catan game governed by dice rolls and hidden card draws), the network is forced to collapse a complex, multimodal probability distribution into a single mean scalar. Under repeated gradient updates on correlated data (the continuous lineage problem), the network overfits to the specific Monte Carlo realizations of the training window rather than learning the true expected value. The failure of the 91-million parameter scaling probe (blowing up on epoch 2) is a textbook manifestation of this regression collapse; larger models memorize the stochastic noise of the scalar targets far faster than smaller models, destroying the value estimates needed for MCTS.   

1.2 The Illusion of Continuous Expert Iteration Under Rigid Gating
The decision to implement a KataGo-style continuous training flywheel was architecturally sound, but mating it to a discrete, high-threshold (+30 Elo) promotion gate is contradictory and self-defeating. The Catan-Zero system has generated a candidate that is statistically stronger (52.8% win rate, equating to approximately +20 Elo) but fails to pass the +30 Elo threshold. By withholding promotion, the system starves the self-play generators of the improved policy.

Expert iteration derives its mathematical power from generating data that is marginally better than the previous generation. Holding a definitively better network back because it did not clear an arbitrary 55% win-rate hurdle halts the core compounding mechanism of AlphaZero-style learning. The rationale that small-compute teams require strict gating to avoid "poisoning" the data pool relies on outdated heuristics. Modern implementations handle this through continuous exponential weight averaging and asynchronous deployment, eliminating the need for hard discrete gates that stall the pipeline entirely.   

1.3 The Hidden Information and Belief State Deficit
The "hidden-information leak" fix applied in the system involved masking the opponent's hidden state and ensuring the network only observed public information. While this prevents the network from cheating, it forces a perfect-information algorithm (AlphaZero) to operate blindly in an imperfect-information domain. AlphaZero assumes a deterministic or fully observable Markov Decision Process. By merely masking the hidden information, the Catan-Zero network treats uncertainty as environmental noise rather than a strategic variable to be inferred and exploited.

In imperfect-information games like Stratego, Poker, or Catan, optimal play requires reasoning about the opponent's private information and taking actions that account for one's own hidden state (e.g., bluffing, hoarding resources for a surprise road-building action). The current architecture lacks any mechanism for belief tracking or counterfactual reasoning. The external transfer gap—where internal Elo gains do not translate equivalently to wins against the hand-tuned catanatron_value bot—is a symptom of the heuristic bot utilizing a superior belief model. The heuristic bot mathematically reconstructs the deck and models opponent hands, whereas Catan-Zero relies entirely on the transformer's latent space to implicitly track cards through the sequence of past public actions, an extraordinarily inefficient use of network capacity.   

1.4 Misallocation of the Gumbel Search Budget
The discovery that the stock mctx implementation of Gumbel MuZero amplifies noise at wide roots via the rescale_to_unit_interval function is a highly significant empirical finding. The minimum-maximum rescaling stretches the narrow value noise of 1-to-2 visit candidates across the entire [0,1] interval, manufacturing false confidence that swamps the near-tied prior.   

The mitigation applied—reducing c_scale from 50.0 to 0.03—effectively neutralizes the completed Q-value's contribution, forcing the Gumbel search to rely almost entirely on the raw policy prior. While this prevents the search from actively losing to its raw policy, it severely neuters the MCTS backup capability. At 54-wide roots with a budget of 64 simulations, a c_scale of 0.03 means the search is functioning as little more than a policy regularizer rather than a deep tactical planner. The failure of the exact-budget sequential halving arm indicates that the interaction between the candidate pruning mechanism and the value estimates is structurally misaligned with the high-branching stochastic nature of early-game Catan.   

2. The Modern Strategic Framework
To break the current plateau and achieve superhuman strength, the system architecture and training pipeline must be modernized to reflect the state-of-the-art in 2024–2026 deep reinforcement learning paradigms. The following solutions directly address the highlighted vulnerabilities.

2.1 Transforming Value Regression into Categorical Classification
The most urgent recommendation is the complete removal of the scalar MSE loss for the value head, replacing it with a categorical cross-entropy loss over discretized bins. This approach, known as HL-Gauss (Histogram Loss with Gaussian smoothing) or Two-Hot encoding, has become the definitive solution for scaling value functions in stochastic environments.   

Instead of predicting a single scalar v∈[−1,1], the value head should project a softmax distribution over k evenly spaced bins (e.g., 128 or 256 bins) spanning the possible return space. The scalar training target z is converted into a categorical target distribution by distributing the probability mass to the adjacent bins corresponding to the true outcome.

Mathematical Mechanism:
If the target value falls between bin z 
i
​
  and bin z 
i+1
​
 , the probability mass is split linearly between them, or smoothed using a Gaussian kernel (HL-Gauss). The network is trained using standard cross-entropy loss against this target distribution:
  

L 
HL
​
 =− 
i=1
∑
m
​
 p 
i
​
 log 
p
^
​
  
i
​
 (x)

During search, the scalar value is recovered simply by taking the expected value of the predicted categorical distribution (the dot product of the softmax probabilities and the bin centers).   

Strategic Rationale:
Categorical cross-entropy acts as implicit label smoothing. It prevents the network from aggressively chasing outlier outcomes (highly common in dice-driven games) and eliminates the exploding gradients that cause value head drift. This shift will immediately resolve the multi-epoch overfitting pathology, allow the safe deployment of the 91M parameter scaling probe, and stabilize continuous learning. The "Stop Regressing" literature explicitly demonstrates that HL-Gauss provides a 1.8–2.1x performance increase in multi-task setups and closes the performance gap in complex board games.   

2.2 Continuous Asynchronous Deployment (EMA) over Hard Gating
The discrete +30 Elo promotion gate must be decommissioned. In the context of a continuous, KataGo-scale compute fleet, rigid gating forces the system into artificial plateaus. The optimal approach is to maintain an Exponential Moving Average (EMA) of the network weights and use this EMA network as the data-generating champion.   

Implementation Mechanics:
Maintain a shadow copy of the network parameters θ 
EMA
​
 . After every training step (or after a fixed number of rows), update the EMA weights:

θ 
EMA
​
 ←βθ 
EMA
​
 +(1−β)θ 
train
​
 

where β is typically 0.99 or 0.999. The generators in the self-play fleet universally pull the latest θ 
EMA
​
  network without a discrete promotion test.   

Validation Protocol:
The external panel of heuristic bots (catanatron_value, AB3) should be run concurrently as a passive telemetry metric to monitor for catastrophic forgetting or inbreeding, rather than as a blocking gate. If the EMA network demonstrates sustained degradation against the external panel, the training window is adjusted or the learning rate is decayed. This architecture allows a true 53% (+20 Elo) improvement to instantly begin generating higher-quality data, compounding the learning process seamlessly rather than trapping the system in endless 600-game "continue/hold" limbo.   

2.3 Bridging the Imperfect Information Gap
Operating purely in the masked-observation regime is insufficient for Catan. While full Counterfactual Regret Minimization (CFR) or Recursive Subgame Solving may be computationally prohibitive given the current Python/Rust per-leaf latency constraints (where a single leaf evaluation on CPU takes 34-38 ms), the system must incorporate Belief-Aware elements.   

Information Set MCTS (IS-MCTS) and Determinization:
Instead of a single MCTS tree that treats hidden state as unknowable noise, the root node should sample N plausible opponent hand compositions (determinizations) consistent with public knowledge. Standard searches are executed on each sampled universe, and the Q-values are aggregated. This technique forces the search to evaluate moves that are robust across multiple possible opponent realities, a requirement for strong imperfect-information play.   

Belief-State Auxiliary Prediction:
To force the shared transformer layers to encode the necessary game-theoretic belief state without slowing down the search, the architecture must incorporate explicit belief heads. An auxiliary classification head attached to the CLS token should be trained to predict the exact distribution of the opponent's resource hand and unplayed development cards. During training, the true unmasked labels (which are banked in the corpus) are used to compute a cross-entropy loss, weighted at approximately 0.25 to prevent it from overwhelming the primary policy/value signals. This mathematically compels the latent space to map public historical actions into private state probability distributions.   

2.4 Architectural Enhancements: Hexagonal Graph Transformers and D6 Equivariance
The current 35M parameter entity-graph transformer discards both the D 
6
​
  (dihedral hexagonal) symmetry and the topological adjacency of the Catan board. Relying on dense attention to rediscover the physical layout of a planar graph is deeply sample-inefficient.   

Symmetry Augmentation:
The finding that D 
6
​
  symmetry is violated is a critical diagnostic. At minimum, rotational and reflectional data augmentation must be explicitly integrated into the training pipeline. Applying one of the 12 dihedral transformations to the board and action labels during batch collation turns a single position into up to 12 distinct training examples, massively accelerating the learning of invariant tactical patterns.   

Graph Biases and Cross-Attention:
Instead of raw dense attention, the transformer should utilize relative positional encodings or attention masking that reflects graph distance on the Catan board (e.g., adjacent intersections have a specific relative embedding, distance-2 intersections have another). Furthermore, scoring actions strictly via cosine similarity to a global CLS token creates an information bottleneck. Action representations should cross-attend directly to the spatial entity tokens (the nodes/hexes they directly affect) before producing logits. The failure of the v3b architecture was highly likely due to insufficient training time and the underlying MSE value-head instability masking the architectural gains; it must be re-evaluated after HL-Gauss is deployed.   

Optimization Vector	Current Implementation	Recommended Architecture	Primary Literature Citation
Value Target Optimization	Mean Squared Error (MSE)	HL-Gauss / Categorical Cross-Entropy	
Farebrother et al. (2024)

Value Transformation	Linear / Min-Max Squash	Symlog (Symmetric Logarithm)	
Hafner et al. (2023)

Flywheel Promotion Logic	Pentanomial SPRT Gate (+30 Elo)	Gateless EMA (Stochastic Weight Averaging)	
Wu / KataGo (2019)

Hidden Information Handling	Masked Public Observation	Belief-State Auxiliary Prediction Heads	
Schmid et al. (2023)

Spatial State Processing	Dense Entity-Token Transformer	D 
6
​
  Symmetry Augmentation + Graph Bias	
Brandstetter et al. (2022)

  
3. Relevant Literature Synthesis (2023–2026)
An examination of recent advancements highlights specific methodologies that solve the exact challenges documented in the Catan-Zero report.

3.1 Scaling Deep RL with Cross-Entropy
The transition from regression to classification for value targets is widely considered the most significant development in deep RL scaling of the past three years. Farebrother et al. (2024) in "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL" conclusively demonstrated that categorical cross-entropy (HL-Gauss) mitigates the noisy-target and non-stationarity issues inherent to bootstrapped value learning. This finding is corroborated by the DreamerV3 architecture, which utilizes "Two-Hot" encoding to represent continuous returns as a probability distribution over discrete buckets, combined with "Symlog" (symmetric logarithm) transformations to compress extreme variance. The implementation of HL-Gauss allows large capacity transformers to scale effectively without the value head dominating or destabilizing the gradients of the shared trunk, explaining precisely why the Catan-Zero 91M probe failed under MSE.   

3.2 Nash-RL, Student of Games, and DeepNash
DeepMind's DeepNash (2022) solved the game of Stratego (a domain with 10 
535
  states and severe imperfect information) without using MCTS at all. Instead, it utilized Regularized Nash Dynamics (R-NaD), a model-free RL algorithm designed to converge to an approximate Nash equilibrium. While converting Catan-Zero to a pure model-free R-NaD architecture would require abandoning the extensive Gumbel MCTS investment, the literature confirms that standard AlphaZero diverges or exploits poorly in high-uncertainty imperfect-information games.   

Alternatively, the "Student of Games" (SoG) framework (Schmid et al., 2023) successfully unified perfect and imperfect information search. SoG utilizes Growing-Tree Counterfactual Regret Minimization (GT-CFR) alongside sound self-play, allowing the search tree to reason about common knowledge and subgame re-solving. The application of Information Set MCTS (IS-MCTS), combined with neural network guidance, has also proven highly effective in modern tabletop games with hidden information (e.g., BattleLore, Big 2, and SkyJo). The consensus in the literature is that some form of belief tracking or counterfactual reasoning is mandatory to bridge the final gap to superhuman performance in these domains.   

3.3 Recent Gumbel Search Advances
The Gumbel MuZero/AlphaZero architecture is specifically designed to guarantee policy improvement under strict simulation constraints. However, the default hyperparameters were tuned for deterministic board games (Chess, Go). Recent adaptations, such as ReSCALE (2026) and Treant-Gumbel, emphasize the importance of Sequential Halving in wide action spaces. The literature suggests that the c_scale sensitivity observed in Catan-Zero is a known artifact of the min-max normalization applied to unvisited nodes. Advanced implementations utilize Variance-Aware UCT or progressive widening to handle chance nodes, preventing the search budget from being squandered on flat probability distributions generated by dice outcomes.   

3.4 Soft-Z and Hybrid Value Bootstrapping
The Catan-Zero report notes that value-target-lambda 0.5—blending the terminal outcome with the search's root value estimate—produced the strongest gate in project history (59.0%). This aligns perfectly with Willemsen et al.'s findings on AlphaZero with greedy backups (A0GB) and Soft-Z targets. Furthermore, utilizing Temporal Difference (TD(λ)) style bootstrapping addresses the sparse terminal reward problem. Lambda-Reachability and similar multi-step estimators interpolate between local self-consistency updates and long-horizon Monte Carlo safety targets, which is critical in 600-decision trajectories where credit assignment decays heavily.   

4. Concrete Experimental Roadmap
The following experimental phases are ranked by expected Return on Investment (ROI), explicitly tailored to the 18 GPU (+ 45 L4 burst) compute envelope and the existing Python/Rust infrastructure.

Phase 1: Halt Value Drift via HL-Gauss
Hypothesis: Replacing scalar MSE with categorical cross-entropy will eliminate value head overfitting, stabilize training across multiple epochs, and permit scaling to the 91M parameter network.

Execution: Modify the value head to output logits for 128 bins spanning [−1,1]. Apply the HL-Gauss projection to the z targets (or the λ=0.5 blended targets). Ensure use_regression = False. Train the 35M network for 3 epochs on the existing static 4.58M row corpus.

Success Criteria: The validation value loss smoothly decreases across all 3 epochs without the catastrophic inflation previously observed. A subsequent n=8 baseline test shows an LLR >2.0 against the MSE baseline.

Phase 2: Unblock the Flywheel via EMA Deployment
Hypothesis: The continuous flywheel is currently blocking incremental (+20 Elo) compounding due to mismatched discrete gate thresholds. EMA checkpointing provides sufficient regression protection to run gateless.

Execution: Initialize an EMA copy of the champion network with a decay rate of β=0.995. Update it at the end of every training round. Instruct the 16-GPU generation fleet to continuously poll and use the EMA network. Remove the pentanomial SPRT gate from the critical generation path.

Success Criteria: Over 48 hours of continuous operation, the anchor telemetry shows steadily improving policy cross-entropy, and the external panel against catanatron_value climbs out of the 40-45% plateau into the >50% range.

Phase 3: Inject Topological Priors (D 
6
​
  Symmetry)
Hypothesis: The network is currently wasting massive capacity learning rotational invariances from scratch. Forcing symmetry via data augmentation will yield an immediate sample efficiency multiplier.

Execution: During data loading on the B200 host, apply a random rotation/reflection (from the hexagonal D 
6
​
  group) to the spatial entity tokens and the corresponding action targets for every batch.

Success Criteria: The evaluation of the prior policy's standard deviation across the 12 symmetric orientations of a fixed board drops from the currently measured 0.175 nats to near zero. Overall internal win-rate improves by 15-20 Elo.

Phase 4: Construct the Belief State Auxiliary Head
Hypothesis: The network cannot effectively plan long-term strategies without implicitly tracking opponent hands. Forcing the representation to encode belief states via auxiliary losses will bridge the gap to the heuristic bots.

Execution: Add an auxiliary classification head to the CLS token designed to predict the exact distribution of the opponent's resource hand and unplayed development cards. Apply a loss weight of 0.25 to this head using the true unmasked labels during training (the labels are banked in the corpus, though masked from the input).

Success Criteria: The internal latent representation clusters distinctly by opponent capability. The external transfer gap closes entirely, resulting in a >55% win rate against the belief-driven catanatron_value bot.

5. Direct Answers to Project Questions (§16)
Q1 — Promotion criterion for the continuous loop.
The rigid +30 Elo SPRT gate designed for discrete generations is fundamentally incompatible with a continuous KataGo-class training loop. The evidence confirms that genuine +20 Elo improvements are being systematically discarded, halting the compounding flywheel. The optimal strategy is to transition to an un-gated EMA deployment. By updating the data-generating fleet with a trailing average of the network weights, the system ensures stability and monotonic improvement without arbitrary statistical blockades. The external panel should serve exclusively as a passive regression tripwire, reverting the EMA weights only upon statistically significant, consecutive external declines.   

Q2 — Escaping the plateau.
The flat anchor telemetry confirms the current data distribution has been fully distilled. Escaping this plateau requires unlocking the generator fleet via the EMA update strategy mentioned above. Furthermore, increasing the topological diversity of the data is paramount. Activating the unwired opponent-pool data (playing 15-20% of generation games against older checkpoints) will force the network out of its narrow self-play stylistic rut, directly addressing potential inbreeding. Increasing the search budget from n=64 to n=96 should only be attempted after stabilizing the value head via HL-Gauss; otherwise, the deeper search will merely amplify miscalibrated scalar noise.   

Q3 — The compression trend.
The compression of gains (+49 → +49 → +33 → +20) is the standard asymptotic curve for self-play reinforcement learning operating within a fixed architectural and search capacity. It indicates that the low-hanging tactical heuristics have been learned. The compression is an artifact of the network exhausting its capacity to model the highly stochastic environment using a constrained MSE scalar target. Transitioning to a categorical value distribution and injecting explicit graph topologies will steepen the improvement curve by radically expanding the model's effective hypothesis space.

Q4 — The external-transfer gap.
The discrepancy between the +150 internal Elo climb and the flatline against the catanatron_value bot is a blaring indicator of self-play inbreeding exacerbated by the imperfect-information deficit. Internal gates inflate because the network learns highly specific countermeasures to its own idiosyncrasies. The heuristic bot operates on a robust Bayesian belief state regarding card tracking. Because Catan-Zero was stripped of omniscience but given no mechanisms for belief inference (such as auxiliary hand-prediction heads or IS-MCTS root sampling), it is playing "blind" against an opponent possessing perfect memory logic. The gap is not merely a statistical anomaly; it is a structural deficiency in game-theoretic modeling.   

Q5 — Value-head fragility.
The fragility is a direct mathematical consequence of using Mean Squared Error (MSE) regression to approximate a bimodal, high-variance stochastic outcome. The mitigations currently deployed (one-dose training, champion-init) are operational band-aids. The structural fix, proven definitively in the 2023–2026 literature, is transforming the value target into a categorical distribution. Implementing HL-Gauss or a DreamerV3-style Two-Hot encoding with Symlog transformations will fundamentally resolve this instability, permitting multi-epoch training and unblocking the 91M scaling probe.   

Q6 — Search at wide stochastic roots.
The validation of c_scale=0.03 confirms that standard Gumbel MuZero min-max normalization acts destructively at wide, low-visit roots by artificially inflating noise. While the 0.03 scaling mitigates the damage, it neuters the MCTS backup. Recent literature introduces ReSCALE and Variance-Aware configurations specifically designed to manage Gumbel sampling in combinatorial spaces without relying on arbitrary c_scale shrinking. Additionally, KataGo-style policy-target pruning (First Play Urgency modifications) should be integrated to heavily suppress the exploration of low-prior candidates at the 54-wide placement nodes, forcing the sequential halving budget into a tighter cluster of viable candidates.   

Q7 — Architecture.
The A/B test failure of the 47.8M v3b architecture is an anomaly of undertraining and MSE collapse, not a refutation of capacity. A dense transformer applied to a sparse, topologically strict hexagonal grid represents a massive waste of parameter efficiency. Graph-biased attention and D 
6
​
 -equivariance are non-negotiable upgrades for superhuman spatial reasoning in board games. Given the leaf-cost economics (where the NN forward pass consumes merely 4% of the latency), scaling the parameter count via grouped convolutions or sparse graph layers is computationally virtually free. The architecture must explicitly encode adjacency to stop forcing the network to relearn the concept of a "road connection" from random self-play.   

Q8 — The Unasked Question: Terminal Reward Density vs. Trajectory Credit Assignment.
The most critical blind spot in the report is the total reliance on a single, delayed scalar terminal reward (Win/Loss) at the end of a ~600-decision stochastic trajectory. In stochastic environments with massive branching factors, credit assignment heavily decays over long horizons. The literature on Temporal Difference learning (TD(λ)) and hybrid objectives indicates that relying solely on Monte Carlo outcomes (±1) starves the network of intermediate tactical signals. The unasked question is: How can the system inject dense intermediate reward signals without destroying the zero-sum theoretical purity? The solution involves applying TD(λ) return tracking to the value targets (blending N-step bootstrapped values with the terminal outcome) and aggressively training the auxiliary Catan-native heads (e.g., longest road, largest army, city count) not just as zero-weighted telemetry, but as dense predictive regularizers that stabilize the lower layers of the transformer.   

