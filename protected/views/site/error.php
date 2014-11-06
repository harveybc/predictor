<?php
$this->pageTitle=Yii::app()->name . ' - Error';
$this->breadcrumbs=array(
	'Error',
);
?>

<?php $this->setPageTitle('Error <?php echo $code; ?>'); ?>

<div class="error">
<?php echo CHtml::encode($message); ?>
</div>