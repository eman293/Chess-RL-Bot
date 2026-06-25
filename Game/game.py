from flask import Flask, jsonify, request
from flask_cors import CORS
from pieces import *
from board import Board, is_in_check, is_in_checkmate, is_stalemate
import copy

app = Flask(__name__)
CORS(app)

global board
last_move = None  
pending_promotion = None  

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
    global board, last_move, pending_promotion
    board = setup()
    last_move = None
    pending_promotion = None
    return jsonify({'board': board.display()})

@app.route('/api/board_to_fen', methods=['GET'])
def board_to_fen():
    global board
    

@app.route('/api/move', methods=['POST'])
def move_piece():
    global last_move, pending_promotion

    data = request.get_json()
    from_row, from_col = data['from']
    to_row, to_col = data['to']

    try:
        piece = board.grid[from_row][from_col]
        if piece == ' ':
            return jsonify({'error': 'No piece at source position'}), 400

        # Simulate move to check it doesn't leave own king in check
        test_board = copy.deepcopy(board)
        test_piece = test_board.grid[from_row][from_col]

        try:
            test_piece.move((from_row, from_col), (to_row, to_col), test_board, last_move)
        except RuntimeError:
            return jsonify({'error': 'Invalid move'}), 400
        except NotImplementedError as e:
            return jsonify({'error': f'Not yet implemented: {str(e)}'}), 400

        if is_in_check(test_board, piece.color, last_move):
            return jsonify({'error': 'Move would leave your king in check'}), 400

        target = board.grid[to_row][to_col]
        if target != ' ' and target.name == 'King':
            return jsonify({'error': 'Cannot capture the king'}), 400

        # Promotion
        if isinstance(piece, Pawn) and (to_row == 7 or to_row == 0):
            board.update(from_row, from_col, to_row, to_col)
            piece.history.append(((from_row, from_col), (to_row, to_col)))
            last_move = (piece, (from_row, from_col), (to_row, to_col))
            pending_promotion = (to_row, to_col, piece.color)
            return jsonify({'promotion': True, 'row': to_row, 'col': to_col, 'board': board.display()})

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
    app.run(host='0.0.0.0', port = 5000)