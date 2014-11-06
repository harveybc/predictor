<?php

/**
 * This is the model class for table "aislamiento_acometida".
 *
 * The followings are the available columns in table 'aislamiento_acometida':
 * @property integer $Toma
 * @property string $TAG
 * @property string $Fecha
 * @property double $A050
 * @property double $A1
 * @property double $B050
 * @property double $B1
 * @property double $C050
 * @property double $C1
 * @property integer $OT
 * @property integer $Estado
 * @property integer $Observaciones
 */
class Aislamiento_acometida extends CActiveRecord
{
        // Nuevas columnas
        //public $Observaciones;
       /**  
	 * Returns the static model of the specified AR class.
	 * @return Aislamiento_acometida the static model class
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
		return 'aislamiento_acometida';
	}

	/**
	 * @return array validation rules for model attributes.
	 */
	public function rules()
	{
		// NOTE: you should only define rules for those attributes that
		// will receive user inputs.
		return array(
			array('Estado, Toma, OT', 'numerical', 'integerOnly'=>true),
			array('A025,A2,B025,B2,C025,C2,A050, A1, B050, B1, C050, C1', 'numerical'),
			array('TAG', 'length', 'max'=>50),
                        array('Estado', 'numerical', 'integerOnly'=>true),
                        array('Observaciones', 'length', 'max'=>250),
			array('Fecha', 'safe'), 
			// The following rule is used by search().
			// Please remove those attributes that should not be searched.
			array('Estado, Observaciones,Toma, TAG, Fecha, A050, A1, B050, B1, C050, C1, OT', 'safe', 'on'=>'search'),
                        array('OT, A050,B050,C050', 'compare', 'compareValue'=>0,'operator'=>'!=','message'=>'Este valor no puede ser 0'),  
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
                    
                    'relacMotores' => array(self::BELONGS_TO, 'Motores', '','on'=>'TAG=TAG'),
		);
	}

	/**
	 * @return array customized attribute labels (name=>label)
	 */
	public function attributeLabels()
	{
		return array(
			'Toma' => 'Toma',
			'TAG' => 'Tag',
			'Fecha' => 'Fecha',
			'A050' => 'A050',
			'A1' => 'A1',
			'B050' => 'B050',
			'B1' => 'B1',
			'C050' => 'C050',
			'C1' => 'C1',
			'OT' => 'Orden de Trabajo',
                        'Estado' => Yii::t('app', 'Estado'),
                        'Observaciones' => Yii::t('app', 'Observaciones'),            
                    
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

		$criteria->compare('Toma',$this->Toma);
		$criteria->compare('TAG',$this->TAG,true);
		$criteria->compare('Fecha',$this->Fecha,true);
		$criteria->compare('A050',$this->A050);
		$criteria->compare('A1',$this->A1);
		$criteria->compare('B050',$this->B050);
		$criteria->compare('B1',$this->B1);
		$criteria->compare('C050',$this->C050);
		$criteria->compare('C1',$this->C1);
		$criteria->compare('OT',$this->OT);

                $criteria->compare('Estado', $this->Estado);
                $criteria->compare('Observaciones', $this->Observaciones);
		return new CActiveDataProvider($this, array(
			'criteria'=>$criteria,
		));
	}
}