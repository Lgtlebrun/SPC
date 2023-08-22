function save_real_gbs(filename,fieldname,field)

for i = 1:size(field,1)
    fid = fopen(filename,'a');
    str{1}=[fieldname,' = '];  % writes the name of the field and the row number
    fprintf(fid,'%s\t',str{1});
    fprintf(fid,'%15.14f\n',field);
    fclose(fid);
end
