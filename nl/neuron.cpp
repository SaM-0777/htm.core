
/**
 * Neurons fires stochastically first
 * As they try to connect to other neurons (maybe neighbour or far away neuron)
 * They connect randomly first to other neurons to form random networks
 *
 * A neuron has many dendrites (inputs) & single axon (output)
 *
 * @property
 * Spike: a stochastic process that generates spikes randomly 1.
 * Connections: A neruon can connect to multiple other neurons
 *
 *
 */

struct Connection {};

class Neuron {
  void constructor() {

  };

public:
  bool state = false;

private:
  /**
   * A stochastic process that generates spikes randomly
   * After spike, it goes through a cool-down time; when it can't spike again
   * Hence the process depends on time
   *
   *
   */
  void spike() {
	
  };
};