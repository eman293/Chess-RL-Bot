from flask import Flask, jsonify, request
from flask_cors import CORS
from pieces import *
from board import Board, is_in_check, is_in_checkmate, is_stalemate
import copy
import math
from stockfish import Stockfish

app = Flask(__name__)
CORS(app)

global board
last_move = None  
pending_promotion = None
last_pawn_move_or_capture = 0
num_moves_total = 1  
stockfish = None

def setup():
    global board
    board = Board()

    # White pieces on rows 0 and 1
    board.grid[0][0] = Rook("W")
    board.grid[0][1] = Knight("W")
    board.grid[0][2] = Bishop("W")
    board.grid[0][3] = Queen("W")
    board.grid[0][4] = King("W")
    board.grid[0][5] = Bishop("W")
    board.grid[0][6] = Knight("W")
    board.grid[0][7] = Rook("W")
    for i in range(8):
        board.grid[1][i] = Pawn("W")

    # Black pieces on rows 6 and 7
    board.grid[7][0] = Rook("B")
    board.grid[7][1] = Knight("B")
    board.grid[7][2] = Bishop("B")
    board.grid[7][3] = Queen("B")
    board.grid[7][4] = King("B")
    board.grid[7][5] = Bishop("B")
    board.grid[7][6] = Knight("B")
    board.grid[7][7] = Rook("B")
    for i in range(8):
        board.grid[6][i] = Pawn("B")

    return board

@app.route('/api/board', methods=['GET'])
def get_board():
    return jsonify(board.display())

@app.route('/api/reset', methods=['POST'])
def reset_board():
    global board, last_move, pending_promotion, last_pawn_move_or_capture, num_moves_total
    board = setup()
    last_move = None
    pending_promotion = None
    last_pawn_move_or_capture = 0
    num_moves_total = 1  
    # print(f"RESET CALLED - num_moves_total is now {num_moves_total}")
    return jsonify({'board': board.display()})


@app.route('/api/move', methods=['POST'])
def move_piece():
    global last_move, pending_promotion, last_pawn_move_or_capture, num_moves_total
    data = request.get_json()
    from_row, from_col = data['from']
    to_row, to_col = data['to']

    try:
        piece = board.grid[from_row][from_col]
        if piece == ' ':
            return jsonify({'error': 'No piece at source position'}), 405

        # Simulate move to check it doesn't leave own king in check
        test_board = copy.deepcopy(board)
        test_piece = test_board.grid[from_row][from_col]

        try:
            test_piece.move((from_row, from_col), (to_row, to_col), test_board, last_move)
        except RuntimeError:
            return jsonify({'error': 'Invalid move'}), 401
        except NotImplementedError as e:
            return jsonify({'error': f'Not yet implemented: {str(e)}'}), 402
        
        if is_in_check(test_board, piece.color, last_move):
            return jsonify({'error': 'Move would leave your king in check'}), 403

        target = board.grid[to_row][to_col]
        if target != ' ' and target.name == 'King':
            return jsonify({'error': 'Cannot capture the king'}), 404

        # Promotion
        if isinstance(piece, Pawn) and (to_row == 7 or to_row == 0):
            board.update(from_row, from_col, to_row, to_col)
            piece.history.append(((from_row, from_col), (to_row, to_col)))
            last_move = (piece, (from_row, from_col), (to_row, to_col))
            pending_promotion = (to_row, to_col, piece.color)
            return jsonify({'promotion': True, 'row': to_row, 'col': to_col, 'board': board.display(), 'color' : piece.color})

        if(isinstance(piece, Pawn) or board.grid[to_row][to_col] != ' '):
            last_pawn_move_or_capture = 0
        else:
            last_pawn_move_or_capture += 1

        num_moves_total += 1
        print(f"MOVE - num_moves_total is now {num_moves_total}")

        # Normal move
        piece.move((from_row, from_col), (to_row, to_col), board, last_move)
        last_move = (piece, (from_row, from_col), (to_row, to_col))

        opponent = 'B' if piece.color == 'W' else 'W'
        in_check = is_in_check(board, opponent, last_move)
        checkmate = in_check and is_in_checkmate(board, opponent, last_move)

        opponent = 'B' if piece.color == 'W' else 'W'
        in_check = is_in_check(board, opponent, last_move)
        checkmate = in_check and is_in_checkmate(board, opponent, last_move)
        stalemate = not in_check and is_stalemate(board, opponent, last_move)

        print(board_to_fen())
        print(get_eval())
        
        return jsonify({
            'board': board.display(),
            'check': in_check,
            'checkmate': checkmate,
            'stalemate': stalemate,
            'checked_color': opponent if in_check else None,
            'checkmated_color': opponent if checkmate else None,
            'stalemated_color': opponent if stalemate else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/eval', methods=['GET'])
def get_eval_route():
    evaluation = get_eval()
    return jsonify({'evaluation': evaluation})

@app.route('/debug/moves')
def debug_moves():
    global num_moves_total
    return jsonify({'num_moves_total': num_moves_total})

def get_eval():
    global stockfish
    fen = board_to_fen()
    print(fen)
    
    stockfish.set_fen_position(str(fen), do_validation = False)
    val = stockfish.get_evaluation()
    return val

def board_to_fen():
    global board, last_move, last_pawn_move_or_capture, num_moves_total

    fen_format = ""
    for i in range(len(board.grid) - 1, -1, -1):
        count_empty = 0
        for j in range(len(board.grid[i])):
            piece = board.grid[i][j]
            if piece != ' ':
                if(count_empty > 0):
                    fen_format += str(count_empty)
                    count_empty = 0
                fen_format += piece.fen
            else:
                count_empty += 1
    
        if(count_empty > 0):
            fen_format += str(count_empty)
        if(i > 0):
            fen_format += "/"
    
    if(last_move is None or last_move[0].color == 'B'):
        fen_format += " w "
    else:
        fen_format += " b "

    pot_w_rookl = board.grid[0][0].history if isinstance(board.grid[0][0], Rook) and board.grid[0][0].color == 'W' else None
    pot_w_king = board.grid[0][4].history if isinstance(board.grid[0][4], King) and board.grid[0][4].color == 'W' else None
    pot_w_rookr = board.grid[0][7].history if isinstance(board.grid[0][7], Rook) and board.grid[0][7].color == 'W' else None

    if (
        pot_w_king is None or 
        (pot_w_king is not None and len(pot_w_king) > 0) or 
        (pot_w_rookl is None and pot_w_rookr is None) or 
        (pot_w_rookl is not None and len(pot_w_rookl) > 0 and pot_w_rookr is not None and len(pot_w_rookr) > 0)
    ):
        fen_format += "--"
    else:
        if pot_w_rookr is not None and len(pot_w_rookr) == 0:
            fen_format += "K"
        if pot_w_rookl is not None and len(pot_w_rookl) == 0:
            fen_format += "Q"

    pot_b_rookl = board.grid[7][0].history if isinstance(board.grid[7][0], Rook) and board.grid[7][0].color == 'B' else None
    pot_b_king = board.grid[7][4].history if isinstance(board.grid[7][4], King) and board.grid[7][4].color == 'B' else None
    pot_b_rookr = board.grid[7][7].history if isinstance(board.grid[7][7], Rook) and board.grid[7][7].color == 'B' else None

    if (
        pot_b_king is None or 
        (pot_b_king is not None and len(pot_b_king) > 0) or 
        (pot_b_rookl is None and pot_b_rookr is None) or 
        (pot_b_rookl is not None and len(pot_b_rookl) > 0 and pot_b_rookr is not None and len(pot_b_rookr) > 0)
    ):
        fen_format += "--"
    else:
        if pot_b_rookr is not None and len(pot_b_rookr) == 0:
            fen_format += "k"
        if pot_b_rookl is not None and len(pot_b_rookl) == 0:
            fen_format += "q"
    
    fen_format += " "

    if(last_move is None or not isinstance(last_move[0], Pawn)):
        fen_format += "- "
    else:
        last_piece, (from_row, from_col), (to_row, to_col) = last_move
        if abs(from_row - to_row) == 2:
            fen_format += f"{chr(to_col + ord('a'))}{8 - to_row - 1} "
        else:
            fen_format += "- "
    
    fen_format += str(last_pawn_move_or_capture) + " " + str(math.floor(num_moves_total / 2))

    return fen_format

@app.route('/api/promote', methods=['POST'])
def promote_piece():
    global pending_promotion, last_move
    if pending_promotion is None:
        return jsonify({'error': 'No promotion pending'}), 400

    data = request.get_json()
    choice = data.get('piece')
    row, col, color = pending_promotion

    piece_map = {
        'queen':  Queen(color),
        'rook':   Rook(color),
        'bishop': Bishop(color),
        'knight': Knight(color),
    }

    if choice not in piece_map:
        return jsonify({'error': 'Invalid piece choice'}), 400

    board.promote(row, col, piece_map[choice])
    pending_promotion = None

    opponent = 'B' if color == 'W' else 'W'
    in_check = is_in_check(board, opponent, last_move)
    checkmate = in_check and is_in_checkmate(board, opponent, last_move)

    opponent = 'B' if color == 'W' else 'W'
    in_check = is_in_check(board, opponent, last_move)
    checkmate = in_check and is_in_checkmate(board, opponent, last_move)
    stalemate = not in_check and is_stalemate(board, opponent, last_move)

    return jsonify({
        'board': board.display(),
        'check': in_check,
        'checkmate': checkmate,
        'stalemate': stalemate,
        'checked_color': opponent if in_check else None,
        'checkmated_color': opponent if checkmate else None,
        'stalemated_color': opponent if stalemate else None
    })
    
@app.route('/debug/check/<color>')
def debug_check(color):
    results = []
    king_pos = None
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece != ' ' and piece.name == 'King' and piece.color == color:
                king_pos = (r, c)

    opponent = 'B' if color == 'W' else 'W'
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece != ' ' and piece.color == opponent:
                try:
                    test_board = copy.deepcopy(board)
                    test_piece = test_board.grid[r][c]
                    test_piece.move((r, c), king_pos, test_board)
                    results.append(f"THREAT: {piece.name} at ({r},{c}) can reach king at {king_pos}")
                except Exception as e:
                    results.append(f"OK: {piece.name} at ({r},{c}) blocked by {str(e)}")

    return jsonify({'king': king_pos, 'results': results})

@app.route('/debug/grid')
def debug_grid():
    grid = []
    for r in range(8):
        row = []
        for c in range(8):
            p = board.grid[r][c]
            row.append(str(p) if p != ' ' else ' ')
        grid.append(row)
    
    test = copy.deepcopy(board)
    grid2 = []
    for r in range(8):
        row = []
        for c in range(8):
            p = test.grid[r][c]
            row.append(str(p) if p != ' ' else ' ')
        grid2.append(row)
    
    return jsonify({'original': grid, 'deepcopy': grid2})

if __name__ == '__main__':
    setup()
    stockfish = Stockfish(path=r"C:\Users\eman2\Documents\GitHub\Project\Game\stockfish\stockfish-windows-x86-64-avx2.exe")
    stockfish.set_depth(15)
    print(board_to_fen())
    board.display()
    print(get_eval())
    app.run(host='0.0.0.0', port = 5000)