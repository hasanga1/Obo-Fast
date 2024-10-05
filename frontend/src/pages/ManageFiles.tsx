import React, { useState, useEffect } from "react";
import { Container, TextField, Button, Grid, Typography, IconButton, Box } from "@mui/material";
import EditIcon from "@mui/icons-material/Edit";
import DeleteIcon from "@mui/icons-material/Delete";

const ManageFiles: React.FC = () => {
  const [files, setFiles] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async (query: string = "") => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://127.0.0.1:8000/api/material/search/?q=${query}`
      );
      if (!response.ok) throw new Error("Failed to fetch files");
      const data = await response.json();
      setFiles(data);
    } catch (error) {
      console.error(error);
      alert("Error fetching files");
    } finally {
      setLoading(false);
    }
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  const handleSearch = () => {
    fetchFiles(searchQuery);
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm("Are you sure you want to delete this file?")) return;

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/api/material/delete/${id}/`,
        {
          method: "DELETE",
        }
      );
      if (response.status === 204) {
        alert("File deleted successfully");
        setFiles(files.filter((file) => file.id !== id));
      } else {
        throw new Error("Failed to delete file");
      }
    } catch (error) {
      console.error(error);
      alert("Error deleting file");
    }
  };

  const handleEdit = (id: number) => {
    // Implement edit logic here
  };

  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Manage Files
      </Typography>
      <Box mb={3}>
        <TextField
          label="Search"
          variant="outlined"
          value={searchQuery}
          onChange={handleSearchChange}
          fullWidth
        />
        <Button
          variant="contained"
          color="primary"
          onClick={handleSearch}
          disabled={loading}
          style={{ marginTop: 10 }}
        >
          Search
        </Button>
      </Box>
      <Grid container spacing={2}>
        {files.map((file) => (
          <Grid item xs={12} key={file.id}>
            <Box
              display="flex"
              alignItems="center"
              justifyContent="space-between"
              border={1}
              borderRadius={1}
              p={2}
              borderColor="grey.400"
            >
              <Typography>
                {file.file_name} ({file.file_type})
              </Typography>
              <div>
                <IconButton color="primary" onClick={() => handleEdit(file.id)}>
                  <EditIcon />
                </IconButton>
                <IconButton color="error" onClick={() => handleDelete(file.id)}>
                  <DeleteIcon />
                </IconButton>
              </div>
            </Box>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default ManageFiles;
