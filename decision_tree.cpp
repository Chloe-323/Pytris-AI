#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>

using namespace std;


enum tetromino_types{
    I = 3,
    J = 5,
    L = 4,
    O = 6,
    S = 2,
    T = 1,
    Z = 0
};

struct tetromino {
    int type;
    int rotation;
    bool grid[4][4];
};

struct tetromino tetr_I = {
    I,
    0,
    {
        {0, 0, 0, 0},
        {1, 1, 1, 1},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    }
};
struct tetromino tetr_J = {
    J,
    0,
    {
        {0, 1, 0, 0},
        {0, 1, 0, 0},
        {1, 1, 0, 0},
        {0, 0, 0, 0}
    }
};
struct tetromino tetr_L = {
    L,
    0,
    {
        {1, 0, 0, 0},
        {1, 0, 0, 0},
        {1, 1, 0, 0},
        {0, 0, 0, 0}
    }
};
struct tetromino tetr_O = {
    O,
    0,
    {
        {0, 0, 0, 0},
        {0, 1, 1, 0},
        {0, 1, 1, 0},
        {0, 0, 0, 0}
    }
};
struct tetromino tetr_S = {
    S,
    0,
    {
        {0, 0, 0, 0},
        {0, 1, 1, 0},
        {1, 1, 0, 0},
        {0, 0, 0, 0}
    }
};
struct tetromino tetr_T = {
    T,
    0,
    {
        {0, 0, 0, 0},
        {0, 1, 0, 0},
        {1, 1, 1, 0},
        {0, 0, 0, 0}
    }
};
struct tetromino tetr_Z = {
    Z,
    0,
    {
        {0, 0, 0, 0},
        {1, 1, 0, 0},
        {0, 1, 1, 0},
        {0, 0, 0, 0}
    }
};
struct tetromino tetr_null = {
    -1,
    0,
    {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    }
};

tetromino* tetrominos[] = {&tetr_I, &tetr_J, &tetr_L, &tetr_O, &tetr_S, &tetr_T, &tetr_Z};

//empty state
struct state {
    int grid[22][10];
    int score;
    tetromino current;
    tetromino held;
    tetromino next[3];
};
struct state initial_state = {
    {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    },
    0,
    tetr_T,
    tetr_null,
    {tetr_I, tetr_J, tetr_O}
};

struct action {
    int rotation;
    int column;
    bool swap;
    int number;
};
void rotate_tetromino(tetromino &tetr) {
    int new_rotation = (tetr.rotation + 1) % 4;
    tetr.rotation = new_rotation;
    bool new_grid[4][4] = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            new_grid[i][j] = tetr.grid[3 - j][i];
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tetr.grid[i][j] = new_grid[i][j];
        }
    }
}

void print_tetromino(tetromino tetr) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (tetr.grid[i][j]) {
                cout << "X ";
            } else {
                cout << ". ";
            }
        }
        cout << endl;
    }
}

void print_state(state s, int tabs = 0) {
    for (int i = 0; i < 22; i++) {
        for (int j = 0; j < 10; j++) {
            if (s.grid[i][j] == 0) {
                cout << ". ";
            } else {
                cout << "X ";
            }
        }
        cout << endl;
        if(tabs > 0) {
            for(int k = 0; k < tabs; k++) {
                cout << "\t";
            }
        }
    }
}

int grade_state(state &s, int method = 1) {
    if(method == 0){
        return s.score;
    }

    int grade = 0;
    int bumpiness = 0;
    int heights[10];
    int holes = 0;

    //calculate bumpiness
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 22; j++) {
            if (s.grid[j][i]) {
                heights[i] = 22 - j;
                break;
            }
            heights[i] = 0;
        }
    }
    for (int i = 0; i < 10; i++) {
        bumpiness += abs(heights[i] - heights[(i + 1) % 10]);
    }
    //count holes
    for (int i = 0; i < 22; i++) {
        for (int j = 0; j < 10; j++) {
            if (s.grid[i][j] == 0 && heights[j] > 22 - i) {
                holes++;
            }
        }
    }
    switch (method) {
        case 0:
            grade = s.score;
            break;
        case 1:
            grade = s.score - bumpiness - (4 * holes);
            break;
        case 2:
            //score - bumpiness
            grade = s.score - bumpiness;
            break;
        default:
            grade = s.score - bumpiness - (4 * holes);
            break;
    }
    return grade;
}

//TODO: predict loss
state predict_outcome(state s, tetromino tetr, action a) {
    int rotation = a.rotation;
    int column = a.column;
    bool swap = a.swap;
    column -= 1; //adjustment for compatibility with the python code
    if(swap && s.held.type != -1) {
        tetromino temp = tetr;
        tetr = s.held;
        s.held = temp;
    }
    if (tetr.type == I) //Hotfix for I tetromino
        column -= 1;
    state new_state = s;
    //rotate tetromino
    for (int i = 0; i < rotation; i++) {
        rotate_tetromino(tetr);
    }
    //position tetromino over column
    bool tetr_vec[4][10] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };
    //make sure that if the column is out of bounds, it is reduced ONLY IF there are squares in the additional columns
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            if(tetr.grid[i][j] == 1 && (j + column) > 9){
                column -= 1;
                continue;
            }
            if(tetr.grid[i][j] == 1 && (j + column) < 0){
                column += 1;
                continue;
            }
        }
    }
    //fill in tetr_vec
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (tetr.grid[i][j]) {
                tetr_vec[i][j + column] = 1;
            }
        }
    }
    //find lowest row where tetromino can be placed and place it there
    for(int i = 21; i >= 0; i--) {
        //if there is overlap between tetr_vec and these rows, then we can't place the tetromino here
        bool overlap = false;
        for (int k = 0; k < 4; k++){
            for (int j = 0; j < 10; j++) {
                if (i + k >= 22 && tetr_vec[k][j]) {
                    overlap = true;
                    continue;
                }
                if (tetr_vec[k][j]) {
                    if (new_state.grid[i + k][j]) {
                        overlap = true;
                        break;
                    }
                }
            }
        }
        if (!overlap){
            //assert that there is nothing above
            for(int k = 0; k < 4; k++){
                for(int j = 0; j < 10; j++){
                    for(int l = i + k - 1; l >= 0; l--){
                        if(l > 21){
                            continue;
                        }
                        if(tetr_vec[k][j]){
                            if(new_state.grid[l][j]){
                                overlap = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        if (!overlap) {
            for (int j = 0; j < 10; ++j){
                for (int k = 0; k < 4; ++k) {
                    if (tetr_vec[k][j]) {
                        new_state.grid[i + k][j] = 1;
                    }
                }
            }
            new_state.current = tetr;
            break;
        }
    }

    //clear full rows
    int rows_cleared = 0;
    for(int i = 0; i < 22; i++){
        bool full = true;
        for(int j = 0; j < 10; j++){
            if(!new_state.grid[i][j]){
                full = false;
                break;
            }
        }
        if(full){
            rows_cleared += 1;
            for(int j = 0; j < 10; j++){
                new_state.grid[i][j] = 0;
            }
            for(int j = i; j > 0; j--){
                for(int k = 0; k < 10; k++){
                    new_state.grid[j][k] = new_state.grid[j - 1][k];
                }
            }
            for(int k = 0; k < 10; k++){
                new_state.grid[0][k] = 0;
            }
        }
    }

    //Fill in rest of values
    new_state.current = new_state.next[0];
    for (int i = 0; i < 2; i++) {
        new_state.next[i] = new_state.next[i + 1];
    }
    new_state.next[2] = *tetrominos[rand() % 7];
    new_state.score += 1; //not identical to the python code, but fulfills the same goal
    new_state.score += rows_cleared * rows_cleared * 10; //not identical to the python code, but fulfills the same goal
    return new_state;
}

struct action get_action(int action_num) {
    action a;

    //swap
    if(action_num >= 40){
        a.swap = true;
        action_num -= 40;
    } else {
        a.swap = false;
    }

    //rotation
    a.rotation = action_num / 10; //floor division; 0-10 = 0, 11-20 = 1, 21-30 = 2, 31-40 = 3
    if ((action_num % 10) < 5){
        a.column = 5 - action_num % 10;
    } else {
        a.column = action_num % 10; //modulo; it can be 0-9
                                //TODO: double check that this is the same as the python code
    }
    a.number = action_num;
    return a;
}

class DecisionTreeNode {
    public:
        int depth;
        int score;
        float q_value;
        struct state current_state;
        float gamma;
        DecisionTreeNode *parent;
        DecisionTreeNode *children[80];
        vector<DecisionTreeNode *> children_vec;
        int num_children;
        int num_expanded;
        int selection_range;
        int action_num;

        DecisionTreeNode(int depth, struct state s, float gamma, int selection_range, int action_num, DecisionTreeNode *parent) {
            this->depth = depth;
            this->current_state = s;
            this->gamma = gamma;
            this->parent = parent;
            this->score = grade_state(s);
            this->q_value = this->score;
            this->num_children = 0;
            this->num_expanded = 0;
            this->action_num = action_num;
            this->selection_range = selection_range;
            for (int i = 0; i < 80; i++) {
                this->children[i] = nullptr;
            }
            ////////////////////////////////////////////////////////////////////////////////
        }

        float update_q_value(){
            float max_q_value = -1000000;
            for(int i = 0; i < this->num_children; i++){
                if(this->children[i]->q_value > max_q_value){
                    max_q_value = this->children[i]->q_value;
                }
            }
            this->q_value = this->score + this->gamma * max_q_value;
            return this->q_value;
        }
        int generate_children(){
            //do nothing if depth is 0
            if(depth == 0){
                return 0;
            }
            //generate children while avoiding duplicate states
            for(int i = 0; i < 80; i++){
                state predicted_state = predict_outcome(current_state, current_state.current, get_action(i));
                bool is_duplicate = false;
                for(int j = 0; j < 80; j++){
                    if(children[j] == nullptr){
                        continue;
                    }
                    if(predicted_state.grid == children[j]->current_state.grid){
                        is_duplicate = true;
                        break;
                    }
                }
                if(!is_duplicate){
                    children[num_children] = new DecisionTreeNode(depth - 1, predicted_state, gamma, selection_range, i, this);
                    children_vec.push_back(children[num_children]);
                    num_children += 1;
                }
            }
            //sort children by q_value
            sort(children_vec.begin(), children_vec.end(), [](DecisionTreeNode *a, DecisionTreeNode *b) {
                return a->q_value > b->q_value;
            });
            int cutoff = children_vec[selection_range - 1]->q_value;
            for(int i = 0; i < num_children; i++){
                if(children_vec[i]->q_value >= cutoff){
                    children_vec[i]->generate_children();
                    num_expanded += 1;
                }
            }
            return num_children;
        }

        int best_action(){
            int best_action_num = 0;
            float best_q_value = -1000000;
            for(int i = 0; i < num_children; i++){
                if(children[i]->q_value > best_q_value){
                    best_q_value = children[i]->q_value;
                    best_action_num = i;
                }
            }
            return best_action_num;
        }

        bool operator<(const DecisionTreeNode &other) const {
            return this->q_value < other.q_value;
        }
        bool operator>(const DecisionTreeNode &other) const {
            return this->q_value > other.q_value;
        }
};

class DecisionTree{
    public:
        DecisionTreeNode *root;
        int depth;
        int selection_range; //Top n nodes to expand
        float gamma;
        struct state initial_state;
        int num_children;
        DecisionTree(struct state s, int depth, int selection_range, float gamma){
            this->depth = depth;
            this->selection_range = selection_range;
            this->gamma = gamma;
            this->initial_state = s;
            cout << "Generating tree..." << endl;
            clock_t begin = clock();
            this->root = new DecisionTreeNode(depth, s, gamma, selection_range, -1, nullptr);
            this->root->generate_children();
            clock_t end = clock();
            this->num_children = root->num_children;
            cout << "Tree generated in " << double(end - begin) / CLOCKS_PER_SEC << " seconds." << endl;
        }
        int best_action(){
            int best_action = -1;
            float best_q_value = -1000000;
            for(int i = 0; i < this->num_children; i++){
                if(this->root->children[i]->q_value > best_q_value){
                    best_q_value = this->root->children[i]->q_value;
                    best_action = i;
                }
            }
            return best_action;
        }
        void walk_tree(){
            cout << "Walking tree..." << endl;
            clock_t begin = clock();
            DecisionTreeNode *current_node = this->root;
            cout << "Initial state: " << endl;
            print_state(current_node->current_state);
            for(int i = 0; i < this->depth; i++){
                int best_action = current_node->best_action();
                current_node = current_node->children[best_action];
                print_state(current_node->current_state);
                cout << "----------------------------" << endl;
            }
            cout << "Tree walked in " << double(clock() - begin) / CLOCKS_PER_SEC << " seconds." << endl;
        }
        state best_state(){
            state best_state = initial_state;
            DecisionTreeNode *cur_node = root;
            for(int i = 0; i < depth; i++){
                best_state = predict_outcome(best_state, best_state.current, get_action(cur_node->best_action()));
                cur_node = cur_node->children[cur_node->best_action()];
            }
            return best_state;
        }
};


//Old main; plays a game randomly for n moves. Keeping for potential debugging
////////////////////////////////////////////////////////////////////////////////
//  int main(int argc, char *argv[]){
//      srand(time(NULL));
//      state s = initial_state;
//      cout << endl;
//      cout << "Initial state:" << endl;
//      print_state(s);
//      cout << endl;
//      state s2 = initial_state;
//      for(int i = 0; i < 80; i++){
//          s2 = predict_outcome(s, s.current, get_action(i));
//  //        s = predict_outcome(s, *tetrominos[rand() % 7], {rand() % 4, rand() % 10, false, 0});
//          cout << "Move " << i << ":" << endl;
//          print_state(s2);
//          cout << "Score: " << s2.score << endl;
//          cout << "Grade: " << grade_state(s2) << endl;
//          cout << endl;
//      }
//      return 0;
//  }
////////////////////////////////////////////////////////////////////////////////


int main(int argc, char *argv[]){
    srand(time(NULL));
    if(argc != 3){
        cout << "Usage: ./main <depth> <selection_range>" << endl;
        return 0;
    }
    int depth = atoi(argv[1]);
    int selection_range = atoi(argv[2]);
    DecisionTree d = DecisionTree(initial_state, depth, selection_range, 0.5);
    d.walk_tree();
}
