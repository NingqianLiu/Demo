<h1 align="center">Quant-Native Deep Learning Framework Demo</h1>

<p align="center">nliu12@illinois.edu &middot; Personal-use licensed</p>

<br>

<h4>Key third-party review section (Claude CLI/Fable 5 Max):</h4>

<p align="center">
  <img src="assets/review-claude-key.jpg" alt="Key third-party review section from Claude CLI/Fable 5 Max" width="88%">
</p>

<p>The full third-party review is attached at the end.</p>

<br>

<h4>Key third-party review section (ChatGPT/GPT-5.6-sol Ultra):</h4>

<p align="center">
  <img src="assets/review-chatgpt-key.jpg" alt="Key third-party review section from ChatGPT/GPT-5.6-sol Ultra" width="88%">
</p>

<p>The full third-party review is attached at the end.</p>

<br>

<h2>Minimal Example</h2>

<ul>
  <li>Use the framework to build modules before constructing the model.</li>
</ul>

<p align="center">
  <img src="assets/module-l.png" alt="Module_L implementation example" width="96%">
</p>

<p>Module_L is a basic module for measuring the magnitude of market moves, both upward and downward. The core logic has three steps: first, compute the log_diff at the minimum kline close resolution; second, apply a WMA to the log_diff series; and third, take the rank of the WMA to obtain its percentile in the historical distribution.</p>

<br>

<h2>Framework Design</h2>

<p><strong>1)</strong> This framework makes quant research iteration really fast. A researcher just builds the model the way an AI engineer builds an AI model, and the pipeline handles the rest: it picks the parameters automatically, then runs the out-of-sample test. What you get back are the parameter sets the optimizer chose that still hold up out-of-sample. So before AI became widespread, in a single day, a researcher can build a strategy with a completely different architecture, or spin through many versions with small tweaks to the model. That takes a strategy from research to deployment in 1 to 3 weeks.</p>

<p><strong>2)</strong> Using this deep-learning-style framework, the optimizer performs hundreds of millions of backtest iterations, with each iteration covering roughly 300 to 500 million time steps. This gives the model substantial fitting capacity while still allowing classical statistical methods to be applied during the evaluation stage to control overfitting.</p>

<p><strong>3)</strong> We define the theoretical limit of backtesting engine used by optimizer as follows: we first run a slow backtest to obtain all filled orders. We then store these orders along with its timestamp, price, size, and direction in an array. Next, we use C++ iterate through the array and compute the PnL for each order. The total time required for this final pass of computing PnL is treated as the theoretical lower bound of computation of backtesting engine, which directly determines the optimizer’s available throughput in the framework.</p>

<p><strong>4)</strong> We designed the backtesting engine to produce results without bias but with variance. The reason is that our optimizer can tolerate variance during optimization, but it cannot tolerate bias. In this setting, bias means that the expected PnL of simulated buy and sell orders systematically differs from the true value. Variance means that the simulated PnL fluctuates randomly around the true value, while its expectation remains correct.</p>

<p><strong>5)</strong> I wrote this deep learning framework because TensorFlow/PyTorch are not naturally adapted to model optimization methods outside gradient-based optimization methods.</p>

<p>I think that although gradient descent or gradient ascent optimization methods can achieve generality for differentiable operators, they are still far from fully releasing the potential of a model.</p>

<p>I think the evaluation criteria for whether an optimizer is good should be: under the premise that there is no style shift between in-sample and out-of-sample data, how well it fits the objective in-sample, and how large the gap is between out-of-sample and in-sample performance. These two mutually conflicting evaluation criteria are the standards for judging whether an optimizer is suitable for this type of data.</p>

<p>Different types of data are suitable for different broad categories of optimizers. For example, financial market data, which has a relatively lower signal-to-noise ratio, and natural language data, which has a relatively higher signal-to-noise ratio, are suitable for different broad categories of optimizers. The signal-to-noise ratio of financial market data can even become arbitrarily close to zero.</p>

<p>The developers’ abstraction of operators in TensorFlow and PyTorch is very advanced, but their abstraction of optimization methods is not advanced enough. They cannot naturally adapt to statistical-method-based optimizers, which prevents these two deep learning frameworks from being used in more general domains.</p>

<p>When using the same reinforcement learning optimization objective, by using gradient-based optimizers and statistical-method-based optimizers, one can often find that, from the perspective of gradient-based optimizers, the parameters produced by statistical-method-based optimizers are still far from gradient convergence. They are usually located on a steep gradient, where the absolute value of the gradient is very high.</p>

<p>In my view, for data whose signal-to-noise ratio approaches zero, gradient-based optimizers usually have a much larger in-sample versus out-of-sample gap than statistical-method-based optimizers. When statistical-method-based optimizers are used properly, even if the data’s signal-to-noise ratio approaches zero, the in-sample and out-of-sample fitted objectives can be very close, and may even show no statistically significant difference.</p>

<p>This became one of the guiding principles behind my design of this framework: as long as this principle is not violated, the framework can be designed in any way and still be workable.</p>

<p><strong>6)</strong> I designed a backtesting framework that sustains 10 to 30 billion bar-equivalent or tick-equivalent events per second on a single physical core (the average Nasdaq-100 name produces about 60 ITCH messages per second, one name's full year replays in about 20 ms on a single core), roughly 1000x faster than many high-performance backtesters. The speedup is not free. To reach this throughput, the engine approximates the matching path rather than reproducing it exactly, so every result carries a small error. This demands a research paradigm adapted to both the advantage and the disadvantage, and that is the paradigm this framework is built around, which was the hardest part. It also shifts where the edge comes from: a good strategy or factor now depends more on the framework than on the individual researcher using it.</p>

<br>

<h2>Demo strategies built with this framework</h2>

<p><strong>1)</strong> I built this strategy in August 2025, took me just a few days. It's probably the simplest strategy out there, and it still makes money. Basically just place orders below market price and wait to get filled. I ran it from August to December 2025, then shut down all my strategies at the end of the year. Each step in the live-trading chart below is one hour.</p>

<p align="center">
  <img src="assets/strategy1-live-nav.png" alt="Live-trading NAV curve for the limit-order strategy" width="96%">
</p>

<p>At step 1750, the loss came from a model flaw. The market jumped 4% within a short window, my model placed orders, and then the price reverted to its prior level shortly after, causing a loss.</p>

<p>At step 900, the drop was caused by a sharp price decline that triggered the strategies’ stop-loss mechanisms, which limits the maximum loss and profit for each trade. This was not a model flaw and cannot be filtered out.</p>

<p>From step 2000 to 2500, I overlooked this cloud instance, which caused the disk to fill up and all the programs to crash.</p>

<p>Backtest log-NAV curve for this model:</p>

<p align="center">
  <img src="assets/strategy1-backtest-nav.png" alt="Backtest log-NAV curves for the limit-order strategy" width="100%">
</p>

<p>For the backtest log-NAV curve, I selected five sub-strategies for live trading, roughly steps 0–335 are in-sample, and steps 335–900 are out-of-sample (40%/60% split), which differs from the common 70%/30% split, because 335 days data is sufficient to train the model, and a longer out-of-sample window improves statistical confidence. The backtest sub-strategies’ edge metric is about 0.23-0.32, defined as sum(24h log-return loss) / sum(24h log-return change), similar to RSI.</p>

<br>

<p><strong>2)</strong> This one is a trend following strategy. I only ran it for about 3 months, because I allocated too much capital about 50 days (1,200 steps), the alpha basically faded out. Before that, the deviation between live trading return and expected return stayed within ±10%.</p>

<p align="center">
  <img src="assets/strategy2-live-nav.png" alt="Live-trading NAV curve for the trend-following strategy" width="96%">
</p>

<p align="center">
  <img src="assets/strategy2-live-deviation.png" alt="Deviation between live-trading and expected return for the trend-following strategy" width="96%">
</p>

<p>Live-trading nav curve when I deployed too much capital:</p>

<p align="center">
  <img src="assets/strategy2-live-nav-overallocated.png" alt="Live-trading NAV curve for the trend-following strategy after over-allocating capital" width="96%">
</p>

<p align="center">
  <img src="assets/strategy2-live-deviation-overallocated.png" alt="Deviation between live-trading and expected return after over-allocating capital" width="96%">
</p>

<p>Backtest log-NAV curve for this trend-following strategy:</p>

<p align="center">
  <img src="assets/strategy2-backtest-nav.png" alt="Backtest log-NAV curves for the trend-following strategy" width="100%">
</p>

<p>Roughly steps 0–315 are in-sample, and steps 315–650 are out-of-sample (50%/50% split).</p>

<br>

<h2>Third-party review (Claude CLI/Fable 5 Max)</h2>

<p align="center">
  <img src="assets/review-claude-1.jpg" alt="Third-party review from Claude CLI/Fable 5, page 1" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-2.jpg" alt="Third-party review from Claude CLI/Fable 5, page 2" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-3.jpg" alt="Third-party review from Claude CLI/Fable 5, page 3" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-4.jpg" alt="Third-party review from Claude CLI/Fable 5, page 4" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-5.jpg" alt="Third-party review from Claude CLI/Fable 5, page 5" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-6.jpg" alt="Third-party review from Claude CLI/Fable 5, page 6" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-7.jpg" alt="Third-party review from Claude CLI/Fable 5, page 7" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-8.jpg" alt="Third-party review from Claude CLI/Fable 5, page 8" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-9.jpg" alt="Third-party review from Claude CLI/Fable 5, page 9" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-10.jpg" alt="Third-party review from Claude CLI/Fable 5, page 10" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-11.jpg" alt="Third-party review from Claude CLI/Fable 5, page 11" width="88%">
</p>

<p align="center">
  <img src="assets/review-claude-12.jpg" alt="Third-party review from Claude CLI/Fable 5, page 12" width="88%">
</p>

<br>

<h2>Third-party review (ChatGPT/GPT-5.6-sol Ultra)</h2>

<p align="center">
  <img src="assets/review-chatgpt-1.jpg" alt="Third-party review from ChatGPT/GPT-5, page 1" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-2.jpg" alt="Third-party review from ChatGPT/GPT-5, page 2" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-3.jpg" alt="Third-party review from ChatGPT/GPT-5, page 3" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-4.jpg" alt="Third-party review from ChatGPT/GPT-5, page 4" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-5.jpg" alt="Third-party review from ChatGPT/GPT-5, page 5" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-6.jpg" alt="Third-party review from ChatGPT/GPT-5, page 6" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-7.jpg" alt="Third-party review from ChatGPT/GPT-5, page 7" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-8.jpg" alt="Third-party review from ChatGPT/GPT-5, page 8" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-9.jpg" alt="Third-party review from ChatGPT/GPT-5, page 9" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-10.jpg" alt="Third-party review from ChatGPT/GPT-5, page 10" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-11.jpg" alt="Third-party review from ChatGPT/GPT-5, page 11" width="88%">
</p>

<p align="center">
  <img src="assets/review-chatgpt-12.jpg" alt="Third-party review from ChatGPT/GPT-5, page 12" width="88%">
</p>
