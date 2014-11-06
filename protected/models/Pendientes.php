<?php

/**
 * This is the model class for table "pendientes".
 *
 * The followings are the available columns in table 'pendientes':
 * @property string $id
 * @property integer $revisado
 * @property string $fecha_enviado
 * @property string $fecha_revisado
 * @property string $ruta
 * @property integer $usuario
 */
class Pendientes extends CActiveRecord
{
	/**
	 * Returns the static model of the specified AR class.
	 * @return Pendientes the static model class
	 */
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	/**
	 * @return string the associated database table name
	 */
	public function tableName()
	{
		return 'pendientes';
	}

	/**
	 * @return array validation rules for model attributes.
	 */
	public function rules()
	{
		// NOTE: you should only define rules for those attributes that
		// will receive user inputs.
		return array(
			array('revisado, fecha_enviado, ruta', 'required'),
			array('revisado', 'numerical', 'integerOnly'=>true),
			array('ruta', 'length', 'max'=>256),
			array('fecha_revisado, usuario', 'safe'),
			// The following rule is used by search().
			// Please remove those attributes that should not be searched.
			array('id, usuario, revisado, fecha_enviado, fecha_revisado, ruta, usuario', 'safe', 'on'=>'search'),
		);
	}

	/**
	 * @return array relational rules.
	 */
	public function relations()
	{
		// NOTE: you may need to adjust the relation name and the related
		// class name for the relations automatically generated below.
		return array(
		);
	}

	/**
	 * @return array customized attribute labels (name=>label)
	 */
	public function attributeLabels()
	{
		return array(
			'id' => 'ID',
			'revisado' => 'Revisado',
			'fecha_enviado' => 'Fecha Enviado',
			'fecha_revisado' => 'Fecha Revisado',
			'ruta' => 'Ruta',
			'usuario' => 'Usuario',
		);
	}

	/**
	 * Retrieves a list of models based on the current search/filter conditions.
	 * @return CActiveDataProvider the data provider that can return the models based on the search/filter conditions.
	 */
	public function search()
	{
		// Warning: Please modify the following code to remove attributes that
		// should not be searched.

		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);
                $criteria->compare('revisado',0);
/*		
                $criteria->compare('fecha_enviado',$this->fecha_enviado,true);
		$criteria->compare('fecha_revisado',$this->fecha_revisado,true);
		$criteria->compare('ruta',$this->ruta,true);
		$criteria->compare('usuario',$this->usuario);
*/
		return new CActiveDataProvider($this, array(
			'criteria'=>$criteria,
		));
	}
}