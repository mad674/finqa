# import flwr as fl
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO)

# # Define a custom strategy to ensure correct parameter aggregation
# class CustomFedAvg(fl.server.strategy.FedAvg):
#     def aggregate_fit(self, server_round, results, failures):
#         """
#         Custom aggregation logic to handle parameter shape mismatches.
#         """
#         print(f"\n🔹 Aggregating results for round {server_round}...")

#         # Stop if not enough results are received
#         if len(results) < self.min_fit_clients:
#             print(f"Not enough clients returned results. Expected: {self.min_fit_clients}, Got: {len(results)}")
#             return None

#         try:
#             # Extract parameters and convert them correctly
#             client_weights = [
#                 np.array(fl.common.parameters_to_ndarrays(res.parameters), dtype=object)  
#                 for _, res in results if res.parameters
#             ]

#             if not client_weights:
#                 print("No valid client weights received.")
#                 return None  

#             # DEBUG: Check parameter shapes from each client
#             param_shapes = [tuple(param.shape for param in client) for client in client_weights]
#             print(f" Client Parameter Shapes: {param_shapes}")

#             #  Ensure all client models have the same shape
#             shape_set = {tuple(param.shape for param in client) for client in client_weights}
#             if len(shape_set) > 1:
#                 print(f" Shape mismatch detected among clients: {shape_set}")
#                 return None  

#             # Convert to NumPy array and aggregate
#             aggregated_parameters = [np.mean([client[i] for client in client_weights], axis=0) for i in range(len(client_weights[0]))]

#             # Convert back to Flower parameter format
#             aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)

#         except Exception as e:
#             print(f" Error during aggregation: {e}")
#             return None  

#         return aggregated_parameters, {}

# # Define the strategy for the FL server
# strategy = CustomFedAvg(
#     fraction_fit=1.0,       
#     fraction_evaluate=1.0,  
#     min_fit_clients=2,      
#     min_available_clients=2 
# )

# logging.info("Starting server...")

# # Start the FL server
# if __name__ == "__main__":
#     print("\n🚀 Starting Federated Learning Server...\n")
#     try:
#         fl.server.start_server(
#             server_address="0.0.0.0:8080",
#             config=fl.server.ServerConfig(num_rounds=1),  
#             strategy=strategy
#         )
#     except KeyboardInterrupt:
#         print("\n Server manually stopped.")
#     except Exception as e:
#         print(f"\n Server encountered an error: {e}")
import flwr as fl
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Define a custom strategy with detailed logging
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        print(f"\n🔹 Aggregating results for round {server_round}...")

        if len(results) < self.min_fit_clients:
            print(f"❌ Not enough clients returned results. Expected: {self.min_fit_clients}, Got: {len(results)}")
            return None

        try:
            # Convert Flower parameters to ndarrays
            client_weights = [
                fl.common.parameters_to_ndarrays(res.parameters)
                for _, res in results if res.parameters
            ]

            if not client_weights:
                print("❌ No valid client weights received.")
                return None

            # Log parameter shapes from each client
            param_shapes = [tuple(param.shape for param in client) for client in client_weights]
            print(f"📏 Client Parameter Shapes: {param_shapes}")

            # Check for shape mismatch
            shape_set = {tuple(param.shape for param in client) for client in client_weights}
            if len(shape_set) > 1:
                print(f"❗ Shape mismatch detected among clients: {shape_set}")
                return None

            # Federated averaging (FedAvg)
            aggregated_parameters = [
                np.mean([client[i] for client in client_weights], axis=0)
                for i in range(len(client_weights[0]))
            ]

            # 🧪 Log values of first few parameters (for inspection)
            print(f"\n🧮 Aggregated Parameters Summary (Round {server_round}):")
            for i, param in enumerate(aggregated_parameters):
                print(f"  Layer {i} | Shape: {param.shape} | Sample values: {param.flatten()[:5]} ...")

            # Convert back to Flower format
            aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)

        except Exception as e:
            print(f"🔥 Error during aggregation: {e}")
            return None

        return aggregated_parameters, {}

# Strategy config
strategy = CustomFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_available_clients=2
)

logging.info("Starting server...")

# Start the FL server
if __name__ == "__main__":
    print("\n🚀 Starting Federated Learning Server...\n")
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=1),
            strategy=strategy
        )
    except KeyboardInterrupt:
        print("\n🛑 Server manually stopped.")
    except Exception as e:
        print(f"\n❌ Server encountered an error: {e}")








