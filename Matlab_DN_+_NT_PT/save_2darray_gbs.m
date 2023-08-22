function save_2darray_gbs(filename,fieldname,field,nghost)

for i = 1:size(field,1)
    fid = fopen(filename,'a');
    str{1}=[fieldname,'(',num2str(i-nghost),',',num2str(1-nghost),':',num2str(size(field,2)-nghost),') ='];  % writes the name of the field and the row number
    fprintf(fid,'%s\t',str{1});
    for j = 2:size(field,2)+1
        str{j}=field(i,j-1);
        if j == size(field,2)+1
            fprintf(fid,'%15.14f\n',str{j});
        else
            fprintf(fid,'%15.14f\t',str{j});
        end
    end
    clear str
    fclose(fid);
end
