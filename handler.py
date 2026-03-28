import os
import io
import torch
import requests
import chess.pgn
import numpy as np
from data_objects.game import Game
from encoder.model import Encoder

server_address = 'chess-app-backend.nakul.one'
server_api = f"https://{server_address}/stockfish_eval"

 
def generate_alternative_pgns(game):    
    if not game:
        print("couldn't read game")
        return [], None, None
    
    # Set up board and get moves
    board = game.board()
    moves = list(game.mainline_moves())
    
    # Play through the moves up to just before our target
    for move in moves:
        board.push(move)
    
    # Get all legal moves from this position
    legal_moves = list(board.legal_moves)
    
    # Create a new PGN for each legal move
    result_pgns = []
    move_sans = []
    
    for legal_move in legal_moves:
        # Create a copy of the game up to the target position
        new_game = chess.pgn.Game()
        
        # Copy headers
        for key in game.headers:
            new_game.headers[key] = game.headers[key]
        
        # Mark game as unfinished
        if "Result" in new_game.headers:
            new_game.headers["Result"] = "*"
        
        # Create the move sequence
        node = new_game
        for move in moves:
            node = node.add_variation(move)
        
        # Add our alternative move
        node = node.add_variation(legal_move)
        
        # Convert to PGN string
        new_pgn = io.StringIO()
        exporter = chess.pgn.FileExporter(new_pgn)
        new_game.accept(exporter)
        
        # Save the PGN and the SAN notation of this move
        result_pgns.append(new_pgn.getvalue())
        move_sans.append(board.san(legal_move))

    return result_pgns, move_sans

def process_game(game, prediction_mode = False):
    def create_position_planes(board: chess.Board, positions_seen: set, cur_player: chess.Color) -> np.ndarray:

        def bb_to_plane(bb: int, player: chess.Color) -> np.ndarray:
            binary = format(bb, '064b')
            h_flipped = np.fliplr(np.array([int(binary[i]) for i in range(64)], dtype=np.float32).reshape(8, 8))
            if player:
                return h_flipped
            else:
                return np.flip(h_flipped)
            
        planes = np.zeros((13, 8, 8), dtype=np.float32)
        
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        
        # white pieces (planes 1-6)
        for i, piece_type in enumerate(piece_types):
            bb = board.pieces_mask(piece_type, chess.WHITE)
            planes[i] = bb_to_plane(bb, cur_player)
        
        # black pieces (planes 7-12)
        for i, piece_type in enumerate(piece_types):
            bb = board.pieces_mask(piece_type, chess.BLACK)
            planes[i + 6] = bb_to_plane(bb, cur_player)
        
        # repetition plane (plane 13)
        current_position = board.fen().split(' ')[0]
        if list(positions_seen).count(current_position) > 1:
            planes[12] = 1.0
        
        return planes

    board = chess.Board()
    positions_seen = set()
    positions_seen.add(board.fen().split(' ')[0])
    
    white_moves = []
    black_moves = []
    
    node = game
    while node.next():
        node = node.next()
        move = node.move
        assert(move is not None)
        cur_player = board.turn

        current_planes = create_position_planes(board, positions_seen, cur_player)
        
        board.push(move)
        
        positions_seen.add(board.fen().split(' ')[0])
        
        next_planes = create_position_planes(board, positions_seen, cur_player)
        assert(not (current_planes==next_planes).all())
        # print_planes(next_planes)
        
        move_planes = np.zeros((34, 8, 8), dtype=np.float32)
        
        # first 13 planes (before move)
        move_planes[0:13] = current_planes
        
        # next 13 planes (after move)
        move_planes[13:26] = next_planes
        
        # castling availability (planes 27-30)
        move_planes[26] = float(board.has_queenside_castling_rights(chess.WHITE))
        move_planes[27] = float(board.has_kingside_castling_rights(chess.WHITE))
        move_planes[28] = float(board.has_queenside_castling_rights(chess.BLACK))
        move_planes[29] = float(board.has_kingside_castling_rights(chess.BLACK))
        
        # side to move (plane 31)
        move_planes[30] = 1 if board.turn is chess.WHITE else 0
        
        # fifty move counter (plane 32)
        move_planes[31] = board.halfmove_clock / 100.0
        
        # move time normalized between 0 and 1 (plane 33)
        # change based on time control
        clock_info = node.comment.strip('{}[] ').split()[1] if node.comment else "0:00:30" 
        try:
            minutes, seconds = map(int, clock_info.split(':')[1:])
            total_seconds = minutes * 60 + seconds
            move_planes[32] = min(1.0, total_seconds / 180.0)
        except:
            move_planes[32] = 0.5
        
        # all 1s (plane 34)
        move_planes[33] = 1.0
        
        if board.turn:
            black_moves.append(move_planes)
        else: # chess.BLACK is falsy
            white_moves.append(move_planes)
    
    if (not prediction_mode) and (len(white_moves) < 10 or len(black_moves) < 10):
        return None
    
    white_array = np.stack(white_moves, axis=0)
    black_array = [] if not black_moves else np.stack(black_moves, axis=0)
    
    return white_array, black_array


class EndpointHandler():
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(os.path.join(model_dir, "6_3.pt"), self.device, weights_only=True)
        self.model = Encoder(self.device)
        state_dict = checkpoint['model_state']
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.d = {
                0: self.say_hi,
                1: self.create_user_embedding,
                2: self.ai_move
                }

    def say_hi(self, _data):
        print('entering test endpoint')

        print('exiting test endpoint')
        return {"reply": "hello from inference api!!"}
    
    def create_user_embedding(self, data):
        print('entering create_username endpoint')
        username = data["username"]
        pgn_content = data["pgn_content"]
        games_per_player = data["games_per_player"]

        l = []
        while True:
            game = chess.pgn.read_game(io.StringIO(pgn_content))
            if game is None:
                print("breaking main loop")
                break
            white = game.headers.get("White")
            black = game.headers.get("Black")
            if white == username:
                color = "white"
            elif black == username:
                color = "black"
            else:
                raise Exception
            try:
                arrs = process_game(game)
            except:
                print("skipped")
                continue
            if arrs is None: # skip if less than 10 moves
                print("skipped")
                continue
            if color == "white":
                l.append(arrs[0])
            else:
                l.append(arrs[1])
        if not l: return None

        inputs = np.array([Game(g).random_partial() for g in l[:games_per_player]])
        num_games = min(len(l), games_per_player)

        tensor = torch.tensor(inputs).float().to(self.device)
        with torch.no_grad():
            embeds = self.model(tensor)
            embeds = embeds.view((1, num_games, -1)).to(self.device)
            centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
            centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)
        centroids_incl = centroids_incl.cpu().squeeze(1)
        final_embeds = centroids_incl[0].numpy().tolist()

        print('exiting create_username endpoint')
        return {"reply": final_embeds}
    
    def ai_move(self, data):
        print('entering ai_move endpoint')
        pgn_string = data["pgn_string"]
        color = data["color"]
        player_centroid = data["player_centroid"]

        game = chess.pgn.read_game(io.StringIO(pgn_string)) 
        alternative_pgns, move_sans = generate_alternative_pgns(game)
        game = chess.pgn.read_game(io.StringIO(pgn_string))

        inputs = []
        for alt_pgn in alternative_pgns:
            game_tensors = process_game(chess.pgn.read_game(io.StringIO(alt_pgn)), True)
            game_tensor = game_tensors[0] if color == "white" else game_tensors[1]
            inputs.append(game_tensor)

        tensor = torch.tensor(np.array(inputs)).float().to(self.device)
        with torch.no_grad():
            embed = self.model(tensor)
            embed = embed / torch.norm(embed)

        arr = embed.cpu().numpy()
        similarities = [np.dot(np.array(player_centroid), embed) for embed in arr]
        result = move_sans[np.argmax(similarities)]

        ordered_moves = np.argsort(similarities).tolist()[::-1]
        try:
            board = game.board()
            moves = list(game.mainline_moves())
            
            for move in moves:
                board.push(move)
            response = requests.post(server_api, json={"fen": board.fen()})

            if response.status_code == 400:
                print(response.text)
                print('exiting ai_move endpoint status code before move')
                return {"reply": result}
            best_eval = response.json()["value"]
            best_move = response.json()["best"]
            best_move = chess.Move.from_uci(best_move)
            best_move = board.san(best_move)

            for move in ordered_moves:
                test_board = board.copy()
                test_board.push(board.parse_san(move_sans[move]))
                response = requests.post(server_api, json={"fen": test_board.fen()})
                if response.status_code == 500:
                    print('exiting ai_move endpoint status code after move')
                    return {"reply": best_move}
                eval = response.json()["value"]
                if (color == "white" and (best_eval - eval < 120)) or (color == "black" and (best_eval - eval > -120)):
                    print('exiting ai_move endpoint nice found!')
                    return {"reply": move_sans[move]}
            print('exiting ai_move endpoint all moves are shit!')
            return {"reply": best_move}

        except Exception as e:
            print('error sending to lichess', e)
        print('exiting ai_move endpoint due to exception')
        return {"reply": result}
    
    def __call__(self, data):
        data = data.get("inputs", data)
        return self.d[data["endpoint_num"]](data)