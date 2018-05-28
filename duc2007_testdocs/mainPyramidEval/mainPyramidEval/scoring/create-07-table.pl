#!/usr/bin/perl

die "Must specify basedir\n" unless @ARGV == 1;

$pan_base_dir = shift @ARGV;

open F, "2007_linguistic_quality.table" or die "Cannot open lin. qual. table: $!\n";
while(<F>)
{
 #       setid               peer    1-5    score 
 if(/^(D\d\d\d\d)\s+\S\s+\S\s+(\S+)\s+(\d+)\s+(\d+)\s*$/)
 {
  $quality{$1}{$2}{$3} = $4;
 }
}
close F;
print STDERR "Read in linguistic quality table\n";

open F, "2007_content.table" or die "Cannot open content table: $!\n";
while(<F>)
{
 #       setid                peer    score 
 if(/^(D\d\d\d\d)\s+\S\s+\S\s+(\S+)\s+(\d+)\s*$/)
 {
  $responsiveness{$1}{$2} = $3;
 }
}
close F;
print STDERR "Read in content table\n";

print STDERR "Running scoring...\n";

process_pans($pan_base_dir);

sub process_pans
{
 my $pan_dir = $_[0];
 
 %scores = ();

 print STDERR "Processing pan dir: $pan_dir\n";
 print STDERR "Running: java -jar DUCView-1.4.jar $pan_dir/*.pan\n";
 open SCORES, "java -jar DUCView-1.4.jar $pan_dir/*.pan 2>>java_err.txt |" or die "Error in java: $!\n"; 
 while(<SCORES>)
 {
  if(/-------/)
  {
   if($ignore_next)
   {
    $ignore_next = 0;
   }
   else
   {
    $ignore_next = 1;
    my $filename = <SCORES>;
    chomp $filename;
    print STDERR "\t$filename\n";
    #                        set                             peer
    $filename =~ /^.*\/(D\d\d\d\d)\.M\.250\.[A-Z]\.([0-9A-Z]+)\.pan/;
    $ducview_file = "$1 $2";
#    print STDERR "DUCVIEW FILE: $ducview_file\n";
   }
  }
  elsif(/^Number of unique contributing SCUs:\s+(\d+)/)
  {
   $scores{$ducview_file}{'num SCUs'} = $1;
#    print STDERR "NUM SCUs: $scores{$ducview_file}{'num SCUs'}\n";
  }
  elsif(/^\s*total extra contributors:\s+(\d+)/)
  {
   $scores{$ducview_file}{'repetitions'} = $1;
#    print STDERR "REPETITIONS: $scores{$ducview_file}{'repetitions'}\n";
  }
  elsif(/^Score using .*\s+([\d\.]+)/)
  {
   $scores{$ducview_file}{'modified score'} = $1;
#    print STDERR "MOD SCORE: $scores{$ducview_file}{'modified score'}\n";
  }  
 }
 close SCORES;
 
 my @pans = glob "$pan_dir/*pan";
  
 for my $pan (glob "$pan_dir/*pan") 
 {
  $pan =~ /^.*\/(D\d\d\d\d)\.M\.250\.[A-Z]\.([0-9A-Z]+)\.pan/ or die "Error getting scorekey from: $pan\n";
  $score_key = "$1 $2";
#  print STDERR "SCORE KEY: $score_key\n";
  my ($set_id, $peer_id) = ($1, $2);
  print   "$set_id\t$peer_id\t",
          ($scores{$score_key}{'modified score'}? $scores{$score_key}{'modified score'} : '0'), "\t",
          ($scores{$score_key}{'num SCUs'}? $scores{$score_key}{'num SCUs'} : '0'), "\t",
  ($scores{$score_key}{'repetitions'}? $scores{$score_key}{'repetitions'} : '0'), "\t";
  for (1..5)
  {
   print  "$quality{$set_id}{$peer_id}{$_}\t";
  }
  print   "$responsiveness{$set_id}{$peer_id}\n";
 }
}
