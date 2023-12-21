CREATE DATABASE IF NOT EXISTS myapp;
USE myapp;

CREATE TABLE IF NOT EXISTS renaming_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    original_category VARCHAR(255),
    modified_category VARCHAR(255),
    status VARCHAR(255)
);

