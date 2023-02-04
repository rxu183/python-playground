#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>  
#include <sys/sendfile.h>
#include <stdio.h>

int readfile(char* fname) {

	int fd = -1;
	struct stat fdstat;
	
	fd = open(fname, O_RDONLY);
	if (fd < 0) {
		perror("open");
		exit(EXIT_FAILURE);
	}
	
	if (fstat(fd, &fdstat) == -1) {
		perror("fstat");
		exit(EXIT_FAILURE);
	}

	if (sendfile(STDOUT_FILENO, fd, 0, fdstat.st_size) < 0) {
        perror("write");
		exit(EXIT_FAILURE);
    }
			
	close(fd);
}


int play() {

	int a;
	int b;
	char buffer[010];
	a = 0x41414141;
	b = 0x42424242;

	if (write(STDOUT_FILENO, "For a moment, nothing happened. Then, after a second or so, nothing continued to happen.\n> ", 91) < 0) {
		perror("write");
	}

	sleep(1);

    if (read(STDIN_FILENO, &buffer, 0xC) < 0) {
        perror("read");
    }
	
	if (a == 31337) {
		system(buffer);
	}

	else if (b == 42) {
		readfile("flag.0");
	}

	else if (b == 23) {
		readfile("vuln1.txt");
	}

	else {
		write(STDOUT_FILENO, "So long and thanks for all the fish.\n", 37);
	}

}


int main(int argc, char *argv[]){
	play();
	exit(EXIT_SUCCESS);
}


