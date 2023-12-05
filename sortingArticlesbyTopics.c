#include <stdio.h>
#include <string.h>

#define MAXLENGTH 19150  // the biggest length of an articles (calculated by me)

int main() {
    FILE *input;
    FILE *business;
    FILE *sport;

    char article[MAXLENGTH];
    int c;
    int i;

    // Reading
    input = fopen("dataset/news/Articlestxt.txt", "r");
    business = fopen("dataset/news/business.txt", "w");
    sport = fopen("dataset/news/sport.txt", "w");

    if (input == NULL) {
        perror("Error opening file");
        return 1;
    }

    while ((c = fgetc(input)) != EOF) {


        if(c == '\"'){
            i = 0;
            while ((c = fgetc(input)) != '\"'){
                label:
                article[i] = c;
                i++;
            }
            if((c = fgetc(input)) == ','){
                c = fgetc(input);
                if(c == 'b'){
                    article[i] = '\0';
                    fputs(article, business);
                    fputc('\n', business);
                } else if (c == 's'){
                    article[i] = '\0';
                    fputs(article, sport);
                    fputc('\n', sport);
                }else{
                    goto label;
                }
            }
            else{
                goto label;
            }
        }

    }

    fclose(input);
    fclose(business);
    fclose(sport);
    return 0;
}
