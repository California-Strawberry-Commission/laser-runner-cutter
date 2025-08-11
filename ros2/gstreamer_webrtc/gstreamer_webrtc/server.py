import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
OFFERER_ID = 123
ANSWERER_ID = 456

peers = {}

async def handler(websocket):
    peer_id = None
    try:
        async for message_str in websocket:
            logging.info(f"Received from {websocket.remote_address}: {message_str}")

            if message_str.startswith("HELLO "):
                try:
                    peer_id = int(message_str.split(" ")[1])
                    
                    if peer_id not in [OFFERER_ID, ANSWERER_ID]:
                        logging.error(f"Invalid peer ID: {peer_id}. Must be {OFFERER_ID} or {ANSWERER_ID}.")
                        await websocket.send("ERROR: Invalid peer ID.")
                        continue

                    peers[peer_id] = websocket
                    logging.info(f"Registered peer: {peer_id}. Current peers: {list(peers.keys())}")
                    
                    await websocket.send(f"HELLO_ACK {peer_id}")
                    logging.info(f"Sent HELLO_ACK {peer_id} to {websocket.remote_address}")

                    if peer_id == OFFERER_ID:
                                await websocket.send("SEND_SDP") 
                                logging.info(f"Sent SEND_SDP to offerer {OFFERER_ID} to initiate the offer.")

                except ValueError:
                    logging.error(f"Invalid HELLO message format: {message_str}")
                    await websocket.send("ERROR: Invalid ID in HELLO message")
            
            else:

                try:
                    message_json = json.loads(message_str)
                    
                    target_peer_id = None
                    if peer_id == 123:
                        target_peer_id = 456
                    elif peer_id == 456:
                        target_peer_id = 123
                    
                    if target_peer_id and target_peer_id in peers:
                        target_websocket = peers[target_peer_id]
                        await target_websocket.send(message_str)
                        logging.info(f"Relayed message from {peer_id} to {target_peer_id}")
                    else:
                        logging.warning(f"No target peer ({target_peer_id}) found or connected for relaying message from {peer_id}. Message: {message_str[:50]}...")

                except json.JSONDecodeError:
                    logging.error(f"Received non-JSON message: {message_str[:50]}...")
                except Exception as e:
                    logging.error(f"Error handling message: {repr(e)}")

    except websockets.exceptions.ConnectionClosedOK:
        logging.info(f"Client {websocket.remote_address} (ID: {peer_id}) disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.warning(f"Client {websocket.remote_address} (ID: {peer_id}) disconnected with error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in handler for {websocket.remote_address}: {repr(e)}")
    finally:
        if peer_id in peers and peers[peer_id] == websocket:
            del peers[peer_id]
            logging.info(f"Unregistered peer: {peer_id}. Current peers: {list(peers.keys())}")


async def main():
    logging.info("Starting WebRTC signalling server on ws://0.0.0.0:8081")
    async with websockets.serve(handler, "0.0.0.0", 8081) as server:
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())