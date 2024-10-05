import React, { useEffect, useState } from "react";
import {
  Container,
  Typography,
  Grid,
  Button,
  Box,
  CircularProgress,
  IconButton,
  List,
  ListItem,
  ListItemText,
  TextField,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DeleteIcon from "@mui/icons-material/Delete";

const UploadFiles: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState<boolean>(false);
  const [course, setCourse] = useState<string>(""); // State for course
  const [subject, setSubject] = useState<string>(""); // State for subject
  const [recentFiles, setRecentFiles] = useState<any[]>([]);

  // State for search query
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Filtered files based on search query
  const filteredFiles = recentFiles.filter((file) =>
    file.file_name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFiles([...files, ...Array.from(event.target.files)]);
    }
  };

  const handleFileUpload = async () => {
    if (files.length === 0) return;

    setUploading(true);

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file); // Append file directly
    });
    formData.append("course", course); // Append course
    formData.append("subject", subject); // Append subject

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("File upload failed");
      }

      const result = await response.json();
      console.log(result); // Handle the response data
      alert("Files uploaded successfully!");
      setFiles([]);
      setCourse(""); 
      setSubject(""); 
      fetchRecentFiles();
    } catch (error) {
      console.error("Error uploading files:", error);
      alert("Error uploading files");
    } finally {
      setUploading(false);
    }
  };

  useEffect(() => {
    fetchRecentFiles();
  }, []);

  const fetchRecentFiles = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/");
      if (!response.ok) {
        throw new Error("Failed to fetch recent files");
      }

      const data = await response.json();
      setRecentFiles(data.reverse()); // Show only 5 most recent files
    } catch (error) {
      console.error("Error fetching recent files:", error);
    }
  };

  const handleFileRemove = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const handleDelete = async (fileId: number) => {
    try {
      const response = await fetch(
        `http://127.0.0.1:8000/${fileId}/`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to delete the file");
      }

      alert("File deleted successfully");
      fetchRecentFiles(); // Refresh the list of recent files
    } catch (error) {
      console.error("Error deleting file:", error);
      alert("Error deleting file");
    }
  };

  return (
    <>
      <Container sx={{ marginTop: "10vh" }}>
        <Typography variant="h4" gutterBottom>
          Upload Files
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Button
              variant="contained"
              component="label"
              startIcon={<CloudUploadIcon />}
              fullWidth
            >
              Select Files
              <input type="file" hidden multiple onChange={handleFileChange} />
            </Button>
          </Grid>
        </Grid>
        <Box mt={2}>
          {files.length > 0 && (
            <Box mt={2}>
              <Typography variant="h6">Selected Files</Typography>
              <Grid container spacing={2}>
                {files.map((file, index) => (
                  <Grid item xs={12} key={index}>
                    <Box
                      display="flex"
                      alignItems="center"
                      justifyContent="space-between"
                      border={1}
                      borderRadius={1}
                      p={2}
                      borderColor="grey.400"
                    >
                      <Typography>{file.name}</Typography>
                      <IconButton
                        color="error"
                        onClick={() => handleFileRemove(index)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}
        </Box>
        <Box mt={2}>
          <TextField
            label="Course"
            variant="outlined"
            fullWidth
            margin="normal"
            value={course}
            onChange={(e) => setCourse(e.target.value)}
          />
          <TextField
            label="Subject"
            variant="outlined"
            fullWidth
            margin="normal"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
          />
        </Box>
        <Box mt={2}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleFileUpload}
            disabled={files.length === 0 || uploading}
            fullWidth
          >
            {uploading ? <CircularProgress size={24} /> : "Upload Files"}
          </Button>
        </Box>
      </Container>

      <Container>
        <Box mt={4}>
          <Typography variant="h6" gutterBottom>
            Uploaded Files
          </Typography>
          <TextField
            label="Search Files"
            variant="outlined"
            fullWidth
            margin="normal"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <Box
            sx={{
              maxHeight: 500, // Adjust height for scrollable area
              overflowY: "auto", // Enable vertical scroll
              border: "1px solid #ddd", // Optional: Add border for visual distinction
              borderRadius: 1, // Optional: Add rounded corners
              padding: 2, // Optional: Add padding inside the scrollable area
            }}
          >
            <List>
              {filteredFiles.map((file) => (
                <ListItem
                  key={file.id}
                  sx={{
                    border: "1px solid #ddd",
                    borderRadius: 1,
                    marginBottom: 1,
                  }}
                >
                  <ListItemText
                    primary={file.file_name}
                    secondary={`Uploaded on: ${file.uploaded_at.slice(0, 10)}`}
                  />
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => handleDelete(file.id)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItem>
              ))}
            </List>
          </Box>
        </Box>
      </Container>
    </>
  );
};

export default UploadFiles;
