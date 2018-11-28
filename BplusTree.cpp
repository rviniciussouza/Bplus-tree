 #include <bits/stdc++.h>
using namespace std;

/* 
 * @rvinicius
 * IFNMG - Montes Claros 

Database File Indexing
Implementacao da estrutura Arvore B+ para criar um arquivo de indices de arquivo em disco 
*/

int MAX; //Armazena o grau maximo da arvore
int MIN; //Armazena o grau minimo da arvore

/* Estrutura para representar uma pagina (Nó) */
typedef struct Nodo{
    int pos;
    int num_chaves;
    int *chaves;
    long long *ponteiro;
    bool isleaf;

    Nodo() {}

    Nodo(int n) {
        chaves = new int[n + 1];
        ponteiro = new long long[n];
    }

    Nodo(int n, int idx) {
        pos = idx;
        chaves = new int[n + 1];
        ponteiro = new long long[n];
        num_chaves = 0;
        isleaf = false;
    }
}Nodo;

typedef struct TreeB {
    int num_nos; /* numero de nos da arvore */
    int raiz; /*A posicao da raiz */
    bool raiz_isleaf; /* Se a raiz atual da arvore é uma folha */
    FILE* arq_indices;

    TreeB(const char *nome_arquivo_indices) {
        arq_indices = fopen(nome_arquivo_indices, "w+");
        num_nos = 0;
        raiz = -1;
        raiz_isleaf = false;
    }
}TreeB;

typedef pair<int, int> ii;

/****************************************
 * Funcao para inserir ou atualizar
 * um nodo
 */

void setNodo(TreeB *Arvore, Nodo *nodo) {
    fseek(Arvore->arq_indices, sizeof(Nodo) * nodo->pos, SEEK_SET);
    fwrite(nodo, sizeof(Nodo), 1, Arvore->arq_indices);
}

/***************************************
 * Funcao para ler um nodo do aquivo
 */

Nodo getNodo(TreeB *Arvore, int pos_nodo) {
    Nodo nodo(MAX);
    fseek(Arvore->arq_indices, sizeof(Nodo) * pos_nodo, SEEK_SET);
    fread(&nodo, sizeof(Nodo), 1, Arvore->arq_indices);
    return nodo;
}

/****************************************
 * Funcao para dividir um nodo em caso 
 * de overflow
 */

void splitNodo(TreeB *Arvore, Nodo *no_atual, Nodo *novo_nodo, int j, ii k) {

    for(int i = j + 2; i <= no_atual->num_chaves; ++i) {
        novo_nodo->chaves[++novo_nodo->num_chaves] = no_atual->chaves[i];
        novo_nodo->ponteiro[novo_nodo->num_chaves-1] = no_atual->ponteiro[i-1];
    }

    novo_nodo->ponteiro[novo_nodo->num_chaves] = no_atual->ponteiro[no_atual->num_chaves];                        
    novo_nodo->chaves[++novo_nodo->num_chaves] = k.first;
    novo_nodo->ponteiro[novo_nodo->num_chaves] = k.second;                    
    
    no_atual->num_chaves = j;

    setNodo(Arvore, novo_nodo);
    setNodo(Arvore, no_atual);
}
    
/*************************************
* Tratar caso: underflow de nó folha
*/

void underflowleaf(Nodo *no_atual, Nodo *folha) {
    int j = (folha->num_chaves + no_atual->num_chaves) / 2;
    int k = no_atual->num_chaves - j; /* k = numero de chaves que serao transferidos para a nova folha */
    for(int i = folha->num_chaves; i >= 1; i -= 1) {
        folha->chaves[i + k] = folha->chaves[i];
        folha->ponteiro[i + k - 1] = folha->ponteiro[i-1];
    }

    for(int i = j + 1, k = 1; i <= no_atual->num_chaves; ++i, ++k) {
        folha->chaves[k] = no_atual->chaves[i];
        folha->ponteiro[k-1] = no_atual->ponteiro[i - 1];
    }

    folha->num_chaves += k;
    no_atual->num_chaves -= k;
}

/***********************************
 * int idx_no - indice do nó atual
 * Nodo n - O novo nó a ser inserido
 * int indice_novo - indice do novo nó
 */

ii insertNo(TreeB *Arvore, int idx_no, int indice_novo, int idx) {
    Nodo no_atual = getNodo(Arvore, idx_no);

    /* Se o no_atual é uma folha, a funcao insertNo retorna true, para informa ao pai que precisa inserir mais uma chave */

    if(no_atual.isleaf) {
        Nodo folha = getNodo(Arvore, indice_novo);
        
        if(folha.num_chaves < MIN - 1) {
            underflowleaf(&no_atual, &folha);
        } 

        no_atual.ponteiro[no_atual.num_chaves] = Arvore->num_nos;

        setNodo(Arvore, &folha);
        setNodo(Arvore, &no_atual);

        return ii(folha.chaves[1], indice_novo);
    }

    ii k = insertNo(Arvore, no_atual.ponteiro[no_atual.num_chaves], indice_novo, idx);
    if(k.first > 0) {

        if(no_atual.num_chaves == MAX -1) {
            Nodo novo_nodo(MAX, ++Arvore->num_nos);
            int j = (no_atual.num_chaves + 1) / 2;
            splitNodo(Arvore, &no_atual, &novo_nodo, j, k);
            return ii(no_atual.chaves[j + 1], novo_nodo.pos);
        }

        else {
            no_atual.chaves[++no_atual.num_chaves] = k.first;
            no_atual.ponteiro[no_atual.num_chaves] = k.second;

            setNodo(Arvore, &no_atual);
        }
    }

    return ii(-1,0);
}

/*****************************************
 * Funcao para inserir uma nova raiz
 */

void addRoot(TreeB *Arvore, int p_right, int key) {
    Nodo nova_raiz(MAX, ++Arvore->num_nos);
    nova_raiz.ponteiro[0] = Arvore->raiz;
    nova_raiz.chaves[++nova_raiz.num_chaves] = key;
    nova_raiz.ponteiro[1] = p_right;
    Arvore->raiz = nova_raiz.pos;

    setNodo(Arvore, &nova_raiz);
}

/******************************************
 * Funcao para inserir uma nova pagina 
 */

void insert(TreeB *Arvore, Nodo n) {
    
    /* Se a arvore ainda nao contiver raiz 
       a raiz sera uma folha
    */

    if(Arvore->num_nos - 1 == 0) {
        Arvore->raiz = n.pos;
        Arvore->raiz_isleaf = true;
    }

    /* Se a raiz da arvore é uma folha */
    else if(Arvore->raiz_isleaf) {  
        addRoot(Arvore, n.pos, n.chaves[1]);      
        Arvore->raiz_isleaf = false;
    }

    else {        
        ii k = insertNo(Arvore, Arvore->raiz, n.pos, n.chaves[1]);
        if(k.first > 0)
            addRoot(Arvore, k.second, k.first);
    }
}

/***************************************
 * Funcao para criar a arvore e gerar
 * o arquivo de indices */ 

void buildTreeBplus(const char *k, const char *nome_arquivoDados) {
    sscanf(k, "%d", &MIN);
    MAX = MIN + MIN;

    TreeB *Arvore = new TreeB("arquivo_indice");

    char nome[256];
    FILE* arquivo_dados = fopen(nome_arquivoDados, "r"); 
    if(!arquivo_dados) printf("Arquivo de dados nao encontrado\n");

    else {
        while(1) {
            Nodo folha(MAX, ++Arvore->num_nos);   
            folha.isleaf = true;

            for (int j = 1; j < MAX; ++j) {
                long long pos = ftell(arquivo_dados);
                if(fscanf(arquivo_dados, "%d %s", &folha.chaves[j], nome) == 2) folha.ponteiro[j-1] = pos, folha.num_chaves++;
                else break;
            }

            if(folha.num_chaves == 0) break;
            else {
                setNodo(Arvore, &folha);        
                insert(Arvore, folha);
            } 
        }
    }

    fseek(Arvore->arq_indices, 0, SEEK_SET);
    fprintf(Arvore->arq_indices, "%d", Arvore->raiz);

    fclose(Arvore->arq_indices);
    fclose(arquivo_dados);
}

bool find(FILE* arq_indice, FILE* arq_dados, int pos_atual, int chave_busca) {
    Nodo no_atual;

    fseek(arq_indice, sizeof(Nodo) * pos_atual, SEEK_SET);
    fread(&no_atual, sizeof(Nodo), 1, arq_indice);

    if(no_atual.isleaf) {
        for(int i=1; i <= no_atual.num_chaves; ++i) {
            if(chave_busca == no_atual.chaves[i]) {
                char nome[256]; int x;
                fseek(arq_dados, no_atual.ponteiro[i-1], SEEK_SET);
                fscanf(arq_dados, "%d %s", &x, nome);
                printf("%s\n", nome);
                return true;
            }
        }

        printf("NO\n");
        return true;
    }

    else {
        
        for(int i=1; i <= no_atual.num_chaves; ++i) {
            if(chave_busca < no_atual.chaves[i]) {
                if(find(arq_indice, arq_dados, no_atual.ponteiro[i-1], chave_busca)) return true;
            }

            else if(chave_busca == no_atual.chaves[i]) {
                if(find(arq_indice, arq_dados, no_atual.ponteiro[i], chave_busca)) return true;
            } 
        }

        if(chave_busca > no_atual.chaves[no_atual.num_chaves])
            if(find(arq_indice, arq_dados, no_atual.ponteiro[no_atual.num_chaves], chave_busca)) return true;
    }

    return false;
}

void find_f(char const *arquivo_dados) {
    FILE* arq_dados = fopen(arquivo_dados, "r");
    FILE* arq_indice = fopen("arquivo_indice", "r");

    int root;
    fscanf(arq_indice, "%d", &root);

    int busca;
    while(scanf("%d", &busca) != EOF) {
        find(arq_indice, arq_dados, root, busca);
    }

    fclose(arq_dados);
    fclose(arq_indice);
}


int main(int argc, char *argv[]) {
    
    if(argc != 3) printf("Entrada invalida.\n");
    else {
        buildTreeBplus(argv[1], argv[2]);
        find_f(argv[2]);
    }

    return 0;
}
