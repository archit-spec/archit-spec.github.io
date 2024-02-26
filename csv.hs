import Data.Csv
import Text.Parsec

data Person = Person { name :: String, age :: Int } deriving (Show)

parseName :: String -> Maybe String
parseName = try . many1 letter

parseInt :: String -> Maybe Int
parseInt = readMaybe

parsePerson :: String -> Maybe Person
parsePerson line = do
  name <- parseName <* char ','
  age <- parseInt
  pure Person name age

main :: IO ()
main = do
  csv <- readFile "data.csv"
  let records = decode csv :: Either String [Record]
  case records of
    Left err -> putStrLn err
    Right parsed -> mapM_ print (map decodeRecord parsed)
  where
    decodeRecord :: Record -> Maybe Person
    decodeRecord record = case record of
      [field1, field2] -> parsePerson (field1 ++ "," ++ field2)
      _ -> Nothing

