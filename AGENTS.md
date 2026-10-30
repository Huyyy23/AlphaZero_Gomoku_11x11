# Gomoku 11x11 + AlphaZero

Reply to the user in Vietnamese. Code, identifiers, comments and docstrings in English.

## Project goal

Gomoku on an 11x11 board with an AlphaZero agent trained by self-play.

- Play modes: Human vs AI, AI vs AI
- Rule modes: **Normal** (5 or more in a row wins) and **Renju** (restrictions for Black: 3-3, 4-4, overline; a move making exactly 5 is always legal)
- Target: self-play ~1000 games for a first working run (e.g. 20 iterations x 50 games), then scale up

## Architecture

```
gomoku/        Game core. No dependency on AI or UI.
  board.py       11x11 state, play(), history
  rules/         base.py (interface), normal.py, renju.py
  features.py    board -> input tensor (~4 planes; +1 forbidden-move plane for Renju)
az/            AlphaZero
  model.py       small ResNet (5-6 blocks, 64 filters), policy head (121), value head
  mcts.py        PUCT, Dirichlet noise at root, temperature
  selfplay.py    game generation, multiprocessing
  replay_buffer.py   8-fold symmetry augmentation
  trainer.py     loss = policy CE + value MSE + L2
  evaluator.py   arena: new model vs best model
players/       base.py (select_move(board) -> (r, c)), human, random, mcts_pure, az_player
ui/            pygame or web, attaches to the Game Runner
scripts/       train.py, play.py, arena.py
tests/         rules tests, especially Renju
configs/       normal.yaml, renju.yaml
```

## Hard rules

1. **Rules are an interface.** MCTS, players and UI only call `rules.legal_moves(board)`, `rules.is_win(board)`, `rules.is_terminal(board)`. Never hardcode Normal or Renju logic outside `gomoku/rules/`.
2. **Value sign.** A child node's value is from the opponent's view. Negate it when backing up in MCTS. This is the most common bug.
3. **Training target z** is assigned after the game ends, from the perspective of the player to move at each stored position.
4. **Renju forbidden moves** are masked before softmax and are never expanded in MCTS.
5. **Symmetry** (rotate/flip) is valid for both rule sets. Apply it to features and policy together.
6. **Hot path performance.** Win check and legal-move generation run millions of times. Use numpy vectorization, numba if needed. No Python loops over all 121 cells per call if avoidable.
7. Do not mix rule sets in one replay buffer. One model per rule set, or fine-tune Normal -> Renju.

## Default hyperparameters (test run)

```yaml
n_iter: 20
games_per_iter: 50
n_simulations: 200      # raise to 400-800 later
c_puct: 1.5
dirichlet_alpha: 0.15
dirichlet_eps: 0.25
temp_moves: 15
buffer_size: 50000      # before augmentation
train_steps_per_iter: 200
batch_size: 128
arena_win_threshold: 0.55
```

## Build order

Each step must run and pass tests before the next one starts.

1. Game core with Normal rules + tests (horizontal, vertical, both diagonals, draw on full board)
2. CLI game runner with RandomPlayer and HumanPlayer (gives both play modes)
3. `MCTSPure` baseline (rollouts), used as a real opponent for evaluation
4. AlphaZero on Normal. **Validate the pipeline on 7x7 or 9x9 first**, then move to 11x11
5. Evaluate: new model vs `MCTSPure` and vs older checkpoints. Win rate should rise across iterations
6. UI
7. Renju rules + thorough tests, then train or fine-tune

## Testing

- Every rule has a unit test. Renju needs cases for 3-3, 4-4, overline, and the exactly-5 exception.
- MCTS sanity test: with a trivial win-in-1 position, the search must pick the winning move.
- Self-play sanity test: sample count per game == number of moves, z values have alternating sign for a decisive game.
- Run tests before declaring any step done.

## Common pitfalls

- No Dirichlet noise or temperature -> all self-play games identical
- Buffer too small (forgets fast) or too large (dominated by old weak data)
- Starting at 11x11 directly instead of validating on a smaller board
- In Normal Gomoku the first player has a large advantage; take this into account when reading arena win rates

## Working agreement

- Make the smallest change that fits the current build-order step. Do not scaffold later steps early.
- Keep `gomoku/` free of torch imports.
- Checkpoints go to `checkpoints/`, logs to `logs/`. Do not commit them.
- Use the globally installed skills when a task matches one.

## Agent skills

### Issue tracker

Issues và spec sống trong GitHub Issues của repo này, thao tác qua `gh` CLI. Xem `docs/agents/issue-tracker.md`.

### Triage labels

Giữ 5 nhãn triage mặc định: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. Xem `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` ở root + `docs/adr/`. Xem `docs/agents/domain.md`.
