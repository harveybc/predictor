<?php
$this->breadcrumbs=array(
	'Vibraciones'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de registros', 'url'=>array('index')),
	array('label'=>'Nuevo Registro', 'url'=>array('create')),
	array('label'=>'Detalles de registro', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Registros', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=Vibraciones::model()->findByAttributes(array('TAG'=>$model->TAG));
if (isset($modelTMP->TAG))
        $nombre=$modelTMP->TAG;
if (isset($modelTMP->TAG))
        $nombre=$nombre.' ('.$modelTMP->Fecha.')';

?>
<?php $this->setPageTitle (' Actualizar Registro:<?php echo $nombre?>'); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'vibraciones-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
