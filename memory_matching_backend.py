async def run_yolo_game(websocket, game_state_key):
    # ...existing code...
    if game_state.get("pairs_found", 0) >= CARD_COUNT // 2:
        logging.info(f"[{game_state_key.upper()}] Game Finished! All pairs found.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "game_over", "payload": "All pairs found!"})
        game_state["running"] = False
        await asyncio.sleep(1.0)
        break
    # ...existing code...

async def run_color_game(websocket, game_state_key):
    # ...existing code...
    if game_state.get("pairs_found", 0) >= CARD_COUNT // 2:
        logging.info(f"[{game_state_key.upper()}] Game Finished! All pairs found.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "game_over", "payload": "All pairs found!"})
        game_state["running"] = False
        await asyncio.sleep(1.0)
        break
    # ...existing code...
