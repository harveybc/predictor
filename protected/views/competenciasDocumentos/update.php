<?php
$this->breadcrumbs=array(
	'Competencias Documentos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Update'),
);

$this->menu=array(
	array('label'=>'Lista de CompetenciasDocumentos', 'url'=>array('index')),
	array('label'=>'Create CompetenciasDocumentos', 'url'=>array('create')),
	array('label'=>'View CompetenciasDocumentos', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Manage CompetenciasDocumentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Update CompetenciasDocumentos #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'competencias-documentos-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Update')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
